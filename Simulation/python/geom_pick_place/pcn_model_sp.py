'''Implements a neural network model which derives from the paper, ``PCN: Point Completion Network''
   by Yuan, Khot, Held, Metz, and Hebert.'''

# python
import os
import sys
# scipy
from numpy.random import choice
from numpy import array, arange, concatenate, mean, minimum, reshape, squeeze, tile, zeros
# tensorflow
import tensorflow
from tensorflow import keras
# geom_pick_place
import point_cloud
from hand_descriptor import HandDescriptor
from pcn_generator import PcnGenerator

class PcnModelSP():

  def __init__(self, nInputPoints, deviceId, modelFileName = None, pcnArchitecture = True):
    '''Initializes the model, either from random weights or from a file.
    
    - Input nInputPoints: Number of points for the input point clouds.
    - Input deviceId: GPU device to use. Choose -1 for default device.
    - Input modelFileName: Name of the model to load. If not specified, network is initialized with
      random weights.
    - Input pcnArchitecture: If True, uses PCN network architecture. If False, uses PointNetGPD.
      Only applicable of modelFileName is None.
    - Returns a PcnModelGraspPrediction instance.
    '''
    
    # set parameters
    self.nInputPoints = nInputPoints
    
    # set device
    if deviceId >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceId)
    
    # load model if modelFileName is specified
    if modelFileName is not None:
      self.model = self.Load(modelFileName)
      return
    
    # generate network model
    if pcnArchitecture:
      self.model = self.GenerateNetworkModelPcn()
    else:
      self.model = self.GenerateNetworkModelGpd()
  
  def GenerateNetworkModelGpd(self):
    '''Constructs the Tensorflow graph for the PointNetGPD model.
    See "PointNetGPD: Detecting Grasp Configurations from Point Sets" by Liang, Ma, Li, Gorner,
    Tang, Fang, Sun, and Zhang.
    
    - Returns model: Tensorflow model.
    '''
    
    # architecture
    inputs = keras.Input(shape=(self.nInputPoints, 3), dtype=tensorflow.float32)
    h = keras.layers.Conv1D(64, kernel_size = 1)(inputs)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Activation("relu")(h)
    h = keras.layers.Conv1D(128, kernel_size = 1)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Activation("relu")(h)
    h = keras.layers.Conv1D(1024, kernel_size = 1)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.GlobalMaxPooling1D()(h)
    h = keras.layers.Dense(512)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Activation("relu")(h)
    h = keras.layers.Dense(256)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Activation("relu")(h)
    h = keras.layers.Dense(1, activation = "sigmoid")(h)
    
    # compile model
    optimizer = keras.optimizers.Adam()
    model = keras.Model(inputs = inputs, outputs = h)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    
    return model  
  
  def GenerateNetworkModelPcn(self):
    '''Constructs the Tensorflow graph for the PCN model.
    See "PCN: Point Completion Network" by Yuan, Khot, Held, Metz, and Hebert.
    
    - Returns model: Tensorflow model.
    '''
    
    # encoder
    inputs = keras.Input(shape=(self.nInputPoints, 3), dtype=tensorflow.float32)
    h1 = keras.layers.Conv1D(128, kernel_size = 1, activation = "relu")(inputs)
    h1 = keras.layers.Conv1D(256, kernel_size = 1, activation = None)(h1)
    h2 = keras.layers.GlobalMaxPooling1D()(h1)
    h2 = keras.layers.RepeatVector(self.nInputPoints)(h2)
    v = keras.layers.Concatenate()([h1, h2])
    v = keras.layers.Conv1D(512, kernel_size = 1, activation = "relu")(v)
    v = keras.layers.Conv1D(1024, kernel_size = 1, activation = None)(v)
    v = keras.layers.GlobalMaxPool1D()(v)
    
    # decoder (points)
    x = keras.layers.Dense(1024, activation = "relu")(v)
    x = keras.layers.Dense(1024, activation = "relu")(x)
    x = keras.layers.Dense(1, activation = "sigmoid")(x)
    
    # compile model
    optimizer = keras.optimizers.Adam()
    model = keras.Model(inputs = inputs, outputs = x)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    
    return model
    
  def Load(self, modelFileName):
    '''Loads the model (in ./tensorflow/models) with the indicated file name.
    - Input modelFileName: Name of the Keras modle in ./tensorflow/models.
    - Returns model: Keras model.
    '''   
    
    directory = os.getcwd() + "/tensorflow/models/"
    path = directory + modelFileName
    model = keras.models.load_model(path)
    print("Loaded " + path + ".")
    return model
  
  def PredictGrasps(self, cloud, grasps):
    '''TODO'''
    
    # input checking
    if len(grasps) == 0:
      return zeros(0)
    
    # collect clouds in grasp frames
    inputs = []
    for T in grasps:
      desc = HandDescriptor(T)
      X = desc.GetPointsInHandFrame(cloud)
      idx = choice(arange(X.shape[0]), size = self.nInputPoints)
      inputs.append(X[idx, :])
    
    # predict
    return squeeze(self.model.predict(array(inputs)), axis = 1)
    
  def PredictPlaces(self, cloud, places, placePosition):
    '''TODO'''
    
    # input checking
    if len(places) == 0:
      return zeros(0)
      
    # preprocessing
    tiledPlacePosition = tile(reshape(placePosition, (1, 3)), (cloud.shape[0], 1))
    
    # collect clouds in place frames
    inputs = []
    for place in places:
      X = point_cloud.Transform(place, cloud) - tiledPlacePosition
      # need to up/down sample, or can we use the cloud directly?
      if X.shape[0] != self.nInputPoints:
        doReplace = X.shape[0] < self.nInputPoints
        idx = choice(arange(X.shape[0]), size = self.nInputPoints, replace = doReplace)
        X = X[idx, :]
      inputs.append(X)
    
    # predict
    return squeeze(self.model.predict(array(inputs)), axis = 1)
    
  def Save(self, modelFileName):
    '''Saves the model to file (in ./tensorflow/models).
    - Input modelFileName: The model file name, excluding path.
    '''

    directory = os.getcwd() + "/tensorflow/models/"
    if not os.path.isdir(directory):
      os.makedirs(directory)
    
    path = directory + modelFileName
    self.model.save(path)
    print("Saved " + path + ".")
    
  def Test(self, dataDirectory, batchSize):
    '''TODO'''
    
    generator = PcnGenerator(dataDirectory, batchSize, "cloud", "correct")
    loss, accuracy = self.model.evaluate(generator)
    return loss, accuracy
    
  def TestPrecision(self, dataDirectory, batchSize, threshold):
    '''TODO'''
    
    # input checking
    if not isinstance(dataDirectory, str):
      raise Exception("Expected str dataDirectory; got {}".format(type(dataDirectory)))
      
    if not isinstance(batchSize, (int, long)):
      raise Exception("Expected int batchSize; got {}".format(type(batchSize)))
      
    if not isinstance(threshold, float):
      raise Exception("Expected float threshold; got {}".format(type(threshold)))
      
    if threshold < 0 or threshold > 1:
      raise Exception("Expected threshold in [0, 1]; got {}.".format(threshold))
    
    # initialize
    generator = PcnGenerator(dataDirectory, batchSize, "cloud", "correct")
    correct = []
    
    # evaluate predictions above threshold in batches
    for i in xrange(len(generator)):
      
      inputs, correctGroundTruth = generator[i]
      correctProb = squeeze(self.model.predict(inputs), axis = 1)
      correct.append(correctGroundTruth[correctProb >= threshold])
      
      sys.stdout.write("\rTesting precision batch {}/{}.".format(i + 1, len(generator)))
      sys.stdout.flush()
      
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    # compute precision
    return mean(concatenate(correct))
    
  def Train(self, dataDirectory, learningRateSchedule, batchSizeSchedule):
    '''TODO'''

    for i in xrange(minimum(len(learningRateSchedule), len(batchSizeSchedule))):
      print("Epoch {}/{}.".format(i + 1, minimum(len(learningRateSchedule), len(batchSizeSchedule))))
      generator = PcnGenerator(dataDirectory, batchSizeSchedule[i], "cloud", "correct")
      keras.backend.set_value(self.model.optimizer.lr, learningRateSchedule[i])
      self.model.fit(generator, epochs = 1, shuffle = False)
      self.Save("pcn_model_sp_backup.h5")