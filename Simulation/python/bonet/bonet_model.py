'''A wrapper for BoNet, a neural network for object instance segmentation with point clouds, 
   described in "Learning object bounding boxes for 3D instance segmentation on point clouds" by Bo,
   Wang, Clark, Hu, Wang, Markham, and Trigoni.'''

# python
import os
import sys
from time import time
# scipy
from numpy.random import choice
from scipy.spatial import cKDTree
from numpy import arange, argmax, array, max, maximum, min, reshape, squeeze, tile, zeros
# bonet
from main_3D_BoNet import BoNet
from helper_data import Data, Data_Configs
# geom_pick_place

class BonetModel():

  def __init__(self, nInputPoints, nCategories, deviceId, modelFileName = None):
    '''Initialize the network with random parameters or load the parameters from a file.
    
    - Input nInputPoints: The number of 3D points that will be input to and classified by the network.
    - Input nCategories: The maximum number of objects to segment.
    - Input deviceId: If multiple GPUs are on the system, and only one should be used, specify its
      device ID here (from 1 to nDevices). Otherwise, set to a negative integer.
    - Input modelFileName: The name of the cptk file, e.g., bonet_model.cptk, that contains saved
      parameters. If not input, the model is initialized with random paramters.
    '''
    
    # input checking
    if not isinstance(nInputPoints, (int, long)):
      raise Exception("Expected int for nInputPoints; got {}.".format(type(nInputPoints)))
      
    if nInputPoints < 0:
      raise Exception("nInputPoints must be non-negative.")
      
    if not isinstance(nCategories, (int, long)):
      raise Exception("Expected int for nCategories; got {}.".format(type(nInputPoints)))
      
    if nCategories < 0:
      raise Exception("nCategories must be non-negative.")
      
    if not isinstance(nCategories, (int, long)):
      raise Exception("Expected int for nCategories; got {}.".format(type(nCategories)))
      
    if not isinstance(modelFileName, (type(None), str)):
      raise Exception("Expected str for modelFileName; got {}.".format(type(modelFileName)))
    
    # parameters
    self.nInputPoints = nInputPoints
    self.nCategories = nCategories
    self.deviceId = deviceId    
    self.trainingTableHeight = -0.08
    
    # set current device
    if deviceId >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceId)
    
    # initialize BoNet
    self.configs = Data_Configs()
    self.configs.points_cc = 3
    self.configs.ins_max_num = nCategories
    self.configs.train_pts_num = nInputPoints
    self.configs.test_pts_num = nInputPoints    
    self.net = BoNet(self.configs)
    
    # build Tensorflow graph
    if modelFileName is None:
      self.net.build_graph()
    else:
      self.Load(modelFileName)

  def Load(self, modelFileName):
    '''Load the network parameters from modelFileName.
    
    - Input modelFileName: Name of the cptk file to load. (This file need not exist, but files
      with the same name and extension .cptk.index, .cptk.meta, and .cptk.data-00000-of-00001 need
      to exist.)
    '''
    
    directory = os.getcwd() + "/tensorflow/models/"
    path = directory + modelFileName
    self.net.load_session(path, self.deviceId)
    print("Loaded {}.".format(path))

  def Predict(self, cloud, scoreThresh):
    '''Retrieve object instance segmentation of the given scene.
    
    - Input cloud: nx3 numpy array, representing scene.
    - Input scoreThresh: Points with certainty values below this are not given a segmentation label.
    - Returns segmentation: n x nCategories boolean matrix, indicating to which category each point
      belongs. If the certainty of the segmentation is below scoreThresh, the row is all zeros;
      otherwise, each row has a single 1.
    - Returns segCorrectProb: m-element vector with estimated probability each point (with a
      predicted segmentation) is segmented correctly. So segCorrectProb >= scoreThresh.
    - Returns segDistribution: n x nCategories matrix, with BoNet's certainty for each point for
      each category. Note that rows do not necessarily sum to 1, but each value is guaranteed to be
      in [0, 1]. (Can roughly be thought of as a distribution, for each point, over categories.)
    '''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("cloud must be 3D.")
    
    if not isinstance(scoreThresh, (int, long, float)):
      raise Exception("Expected scalar scoreThresh.")
    
    if cloud.shape[0] == 0:
      return zeros((0, self.nCategories), dtype = "bool"), zeros(0, dtype = "float"), \
        zeros((0, self.nCategories), dtype = "float")
    
    # sample partial cloud to fixed size
    idx = choice(arange(cloud.shape[0]), size = self.nInputPoints)
    dsCloud = cloud[idx, :]
    
    # make prediction
    dsSeg, dsSegCorrectProb, dsMask = self.PredictRaw(dsCloud, scoreThresh)
    
    # get segmentation for original cloud
    cloudTree = cKDTree(dsCloud)
    _, i = cloudTree.query(cloud)
    segmentation = dsSeg[i]
    segDistribution = dsMask[i]
    segCorrectProb = dsSegCorrectProb[i]
    
    # return result
    return segmentation, segCorrectProb, segDistribution
    
  def PredictRaw(self, cloud, scoreThresh):
    '''Retrieve object instance segmentation for a cloud with self.nInputPoints points.
    
    - Input cloud: nInputPoints x 3 numpy array. Does not need to be centered.
    - Input scoreThresh: Points with certainty values below this are not given a segmentation label.
    - Returns segmentation: nInputPoints x nCategories binary matrix, where each row indicates the
      category to which each point belongs. If the row is all zeros, no segmentation is provided
      for this point.
    - Returns segCorrectProb: For each nonzero row in segmentation, the certainty value for the
      selected category.
    - Returns pmask: nInputPoints x nCategories matrix, with BoNet's certainty for each point for
      each category. Note that rows do not necessarily sum to 1, but each value is guaranteed to be
      in [0, 1]. (Can roughly be thought of as a distribution, for each point, over categories.)
    '''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("Expected 3D cloud; got shape {}".format(cloud.shape))
      
    if cloud.shape[0] != self.nInputPoints:
      raise Exception("PredictRaw requires cloud with {} points; got {}.".format( \
        self.nInputPoints, cloud.shape[0]))
      
    if not isinstance(scoreThresh, (int, long, float)):
      raise Exception("Expected scalar scoreThresh.")
    
    # 0-1 center (this is done in helper_data.load_fixed_points)
    cloud = cloud.copy()
    min_x = min(cloud[:, 0]); max_x = max(cloud[:, 0])
    min_y = min(cloud[:, 1]); max_y = max(cloud[:, 1])
    min_z = min(cloud[:, 2]); max_z = max(cloud[:, 2])
    cloud[:, 0] = (cloud[:, 0] - min_x) / maximum((max_x - min_x), 1e-3)
    cloud[:, 1] = (cloud[:, 1] - min_y) / maximum((max_y - min_y), 1e-3)
    cloud[:, 2] = (cloud[:, 2] - min_z) / maximum((max_z - min_z), 1e-3)
    
    # forward pass on network
    bbvert, bbscore, pmask = self.net.sess.run([self.net.y_bbvert_pred_raw,
      self.net.y_bbscore_pred_raw, self.net.y_pmask_pred_raw], feed_dict = 
      {self.net.X_pc: array([cloud]), self.net.is_train: False})    
    
    # resize outputs
    bbscore = squeeze(bbscore) # shape = nCategories
    pmask = squeeze(pmask).T # shape = (nInputPoints, nCategories)
    bbvert = squeeze(bbvert) # shape = (nCategories, 2, 3)
    
    # combine mask with bounding box scores
    bbscore = tile(reshape(bbscore, (1, self.nCategories)), (self.nInputPoints, 1))
    pmask = pmask * bbscore
    
    # Normalization. While it makes sense to normalize rows to sum to 1 (the point must belong to
    # one of the categories), this removes information about bbscore.
    
    #pmask = pmask / repeat(reshape(sum(pmask, axis = 1),
    #  (self.nInputPoints, 1)), self.nCategories, axis = 1)
    
    # put into mask form
    segIdx = argmax(pmask, axis = 1)
    rowIdx = arange(segIdx.size)
    segmentation = zeros((cloud.shape[0], self.nCategories), dtype = "bool")
    segmentation[rowIdx, segIdx] = 1
    segCorrectProb = squeeze(pmask[rowIdx, segIdx])
    
    # eliminate points which fall below the score threshold
    idx = max(pmask, axis = 1) < scoreThresh
    segmentation[idx, :] = 0
    
    # return result
    return segmentation, segCorrectProb, pmask

  def Save(self, modelFileName):
    '''Saves the current model with the specified file name in ./tensorflow/models.
    
    - Input modelFileName: String of the file name, excluding path but including extention, .cptk.
      Note that actually 3 files are created/overwritten: [modelFileName].index,
      [modelFileName].meta, and [modelFileName].data-00000-of-00001.
    '''
    
    directory = os.getcwd() + "/tensorflow/models/"
    if not os.path.isdir(directory):
      os.makedirs(directory)
    
    self.net.saver.save(self.net.sess, save_path = directory + modelFileName)
    print("Saved {}.".format(directory + modelFileName))

  def Test(self, dataDirectory, inputName, outputName, batchSize):
    '''Get BoNet's loss on the given dataset. (Not implemented.)'''

    raise Exception("Not implemented.")

  def Train(self, dataDirectory, inputName, outputName, learningRateSchedule, batchSize):
    '''Train BoNet on the given dataset. Prints progress and saves backup file
       "bonet_model_backup.cptk" at the end of every epoch (pass through the data).
    
    - Input dataDirectory: Directory of .mat files containing 2 arrays, 1 with input point cloud,
      each of size self.nInputPoints x 3, and 1 with ground truth segmentation, a binary array of
      size self.nInputPoints x self.nCategories. Both should be of type float32.
    - Input inputName: Name of the input point cloud in each file in dataDirectory.
    - Input outputName: Name of the g.t. segmentation matrix in each file in dataDirectory.
    - Input learningRateSchedule: List with the learning rate to use. The number of epochs to train,
      i.e., the number of passes through the data is len(learningRateSchedule). (Data is randomly
      shuffled before each epoch).
    - Input batchSize: Number of samples to use for estimating the gradient. Original BoNet used 4.
    '''
    
    # input checking
    
    if not isinstance(dataDirectory, str):
      raise Exception("Expected str dataDirectory.")
    if not isinstance(inputName, str):
      raise Exception("Expected str inputName.")
    if not isinstance(outputName, str):
      raise Exception("Expected str outputName.")
    if not isinstance(batchSize, (int, long)):
      raise Exception("Expected integer batchSize.")
    
    # train
    
    net = self.net
    data = Data(self.configs, dataDirectory, inputName, outputName, batchSize)    
    print('total train batch num:', data.total_train_batch_num)    
    
    for epoch, learningRate in enumerate(learningRateSchedule):

      data.shuffle_train_files(epoch)
      epLossSum = 0.0; startTime = time()
      
      for i in range(data.total_train_batch_num):
        
        # training
        bat_pc, bat_bbvert, bat_pmask = data.load_train_next_batch()
        
        _, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask \
          = net.sess.run([net.optim, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce,
          net.bbvert_loss_iou, net.bbscore_loss, net.pmask_loss], feed_dict = 
          {net.X_pc:bat_pc[:, :, 0:9], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask,
          net.lr:learningRate, net.is_train:True})
        
        # print progress
        epLossSum += ls_bbvert_all + ls_bbscore + ls_pmask
        avgEpLoss = epLossSum / (i + 1)
        avgEpTime = (time() - startTime) / (i + 1)
        remTime = (data.total_train_batch_num - (i + 1)) * avgEpTime / 3600
        
        sys.stdout.write("\repoch: {}/{}, batch: {}/{}, loss: {}, time: {} hours".format(
          epoch + 1, len(learningRateSchedule), i + 1, data.total_train_batch_num, avgEpLoss,
          remTime))
        sys.stdout.flush()
      
      sys.stdout.write("\n")
      sys.stdout.flush()
  
      # save backup of model    
      self.Save("bonet_model_backup.cptk")