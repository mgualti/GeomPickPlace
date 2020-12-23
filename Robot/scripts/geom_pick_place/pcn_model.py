'''Implements a neural network model which derives from the paper, ``PCN: Point Completion Network''
   by Yuan, Khot, Held, Metz, and Hebert.'''

# python
import os
# scipy
from numpy.random import choice
from scipy.special import expit
from numpy import array, arange, mean, minimum, repeat, squeeze
# tensorflow
import tensorflow
from tensorflow import keras
from tensorflow.keras import backend
# geom_pick_place
from pcn_generator import PcnGenerator

class PcnModel():

  def __init__(self, nInputPoints, nOutputPoints, errorThreshold, deviceId, modelFileName = None):
    '''Initializes the model, either from random weights or from a file.
    - Input nInputPoints: Number of points for the input point clouds.
    - Input nOutputPoints: Number of points for the predicted, complete point cloud.
    - Input errorThreshold: A completed point is considered correct if it is within Euclidean
      distance errorThreshold of the nearest ground truth point.
    - Input deviceId: GPU device to use. Choose -1 for default device.
    - Input modelFileName: Name of the model to load. If not specified, network is initialized with
      random weights.
    - Returns a PcnModel instance.
    '''
    
    # set parameters
    self.nInputPoints = nInputPoints
    self.nOutputPoints = nOutputPoints
    self.errorThreshold = errorThreshold
    self.trainingTableHeight = -0.08
    
    # set device
    if deviceId >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceId)
    
    # set loss function
    def ChamfarDistance(Y, X):
      '''The Chamfar Distance (CD) loss used in Yuen et al.
      - Input Y: Ground truth labels, with shape (batch, nOutputPoints, 3).
      - Input X: Predicted labels, with shape (batch, nOutputPoints, 4).
      - Returns the CD which is a vector equal to the batch size.
      '''
      
      X = X[:, :, 0:3] # x, y, z
      
      # average distance of predicted points to nearest ground truth point
      YY = backend.tile(Y, (1, self.nOutputPoints, 1))
      XX = backend.tile(X, (1, 1, self.nOutputPoints))
      XX = backend.reshape(XX, (tensorflow.shape(X)[0], self.nOutputPoints**2, 3))
      Z = backend.sqrt(keras.backend.sum(keras.backend.square(YY - XX), axis = 2))
      Z = backend.reshape(Z, (tensorflow.shape(Z)[0], self.nOutputPoints, self.nOutputPoints))
      Z = backend.min(Z, axis = 2)
      l1 = backend.mean(Z, axis = 1)
      
      # average distance of ground truth points to nearest predicted point
      XX = backend.tile(X, (1, self.nOutputPoints, 1))
      YY = backend.tile(Y, (1, 1, self.nOutputPoints))
      YY = backend.reshape(YY, (tensorflow.shape(Y)[0], self.nOutputPoints**2, 3))
      Z = backend.sqrt(backend.sum(backend.square(YY - XX), axis = 2))
      Z = backend.reshape(Z, (tensorflow.shape(Z)[0], self.nOutputPoints, self.nOutputPoints))
      Z = backend.min(Z, axis = 2)
      l2 = backend.mean(Z, axis = 1)
      
      # loss is the sum of the 2
      return l1 + l2
      
    def ErrorLessThanThreshold(Y, X):
      '''Point-wise binary cross-entropy for probability a point is off by less than a threshold.
      - Input Y: Ground truth labels, with shape (batch, nOutputPoints, 3).
      - Input X: Predicted labels, with shape (batch, nOutputPoints, 4).
      - Returns the CD which is a vector equal to the batch size.
      '''
      
      E = X[:, :, 3] # error
      X = X[:, :, 0:3] # x, y, z
      
      # 1 if Euclidean distance to nearest g.t. point is less than threshold; 0 otherwise
      YY = backend.tile(Y, (1, self.nOutputPoints, 1))
      XX = backend.tile(X, (1, 1, self.nOutputPoints))
      XX = backend.reshape(XX, (tensorflow.shape(X)[0], self.nOutputPoints**2, 3))
      Z = backend.sqrt(keras.backend.sum(backend.square(YY - XX), axis = 2))
      Z = backend.reshape(Z, (tensorflow.shape(Z)[0], self.nOutputPoints, self.nOutputPoints))
      Z = backend.min(Z, axis = 2)
      Z = backend.less_equal(Z, self.errorThreshold)
      
      # cross entropy loss between actual errors and predicted errors
      return keras.losses.binary_crossentropy(Z, E, from_logits = True)
        
    self.ShapeCompletionLoss = ChamfarDistance
    self.CorrectnessProbabilityLoss = ErrorLessThanThreshold
    
    # load model if modelFileName is specified
    if modelFileName is not None:
      self.model = self.Load(modelFileName)
      return
    
    # generate network model
    self.model = self.GenerateNetworkModel()
    
  def GenerateNetworkModel(self):
    '''Constructs the Tensorflow graph for the PCN model.
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
    x = keras.layers.Dense(self.nOutputPoints * 3, activation = None)(x)
    x = keras.layers.Reshape((self.nOutputPoints, 3))(x)
    
    # decoder (correctness probability)
    p = keras.layers.Dense(1024, activation = "relu", name = "p-Dense1")(v)
    p = keras.layers.Dense(1024, activation = "relu", name = "p-Desne2")(p)
    p = keras.layers.Dense(self.nOutputPoints, activation = None, name = "p-Dense3")(p)
    p = keras.layers.Reshape((self.nOutputPoints, 1))(p)
    x = keras.layers.Concatenate()([x, p])
    
    # compile model
    optimizer = keras.optimizers.Adam()
    model = keras.Model(inputs = inputs, outputs = x)
    model.compile(optimizer = optimizer, loss = self.ShapeCompletionLoss)
    
    return model
    
  def Load(self, modelFileName):
    '''Loads the model (in ./tensorflow/models) with the indicated file name.
    - Input modelFileName: Name of the Keras modle in ./tensorflow/models.
    - Returns model: Keras model.
    '''   
    
    directory = os.getcwd() + "/tensorflow/models/"
    path = directory + modelFileName
    model = keras.models.load_model(path, custom_objects = {
      "ChamfarDistance":self.ShapeCompletionLoss,
      "ErrorLessThanThreshold":self.CorrectnessProbabilityLoss})
    print("Loaded " + path + ".")
    return model
  
  def Predict(self, cloud, tableHeight):
    '''Predicts the completed shape and per-point probability of error.
    - Input cloud: The partial cloud of the object to complete. nx3 numpy array with any n.
    - Input tableHeight: The height of the support surface on which the objects are located.
    - Returns cloud: The completed point cloud, self.nInputPoints x 3.
    - Returns correctnessProbability: The probability the Euclidean distance from each predicted
      point to the nearest ground truth point is less than the threshold used during training.
    '''
    
    # sample partial cloud to fixed size
    idx = choice(arange(cloud.shape[0]), size = self.nInputPoints)
    cloud = cloud[idx, :]
        
    # shift partial cloud (center x and y)
    center = array([mean(cloud, axis = 0)]); center[:, 2] = 0
    # (adjust table height based on table height during training)
    center[:, 2] = tableHeight - self.trainingTableHeight
    cloud -= repeat(center, cloud.shape[0], axis = 0)
    
    # predict
    cloud = self.model.predict(array([cloud]))
    cloud = squeeze(cloud)
    correctProb = expit(cloud[:, 3])
    cloud = cloud[:, 0:3]
    
    # shift completed cloud back
    cloud += repeat(center, cloud.shape[0], axis = 0)
    
    # return result
    return cloud, correctProb
    
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
    
  def TestCompletion(self, dataDirectory, batchSize):
    '''Evaluates Chamfer distance over entire dataset.
    - Input dataDirectory: The directory containing the dataset (.mat files with g.t. completions).
    - Input batchSize: The number of examples to evaluate in parallel.
    - Returns: Average Chamfer distance.
    '''
    
    self.model.compile(optimizer = self.model.optimizer, loss = self.ShapeCompletionLoss)
    generator = PcnGenerator(dataDirectory, batchSize)
    return self.model.evaluate(generator)
    
  def TestCorrectnessProbability(self, dataDirectory, batchSize):
    '''Evaluates the cross-entropy loss for the correctness probability prediction.
    - Input dataDirectory: The directory containing the dataset (.mat files with g.t. completions).
    - Input batchSize: The number of examples to evaluate in parallel.
    - Returns: Average binary cross-entropy for correctness probability predictions.
    '''
    
    self.model.compile(optimizer = self.model.optimizer, loss = self.CorrectnessProbabilityLoss)
    generator = PcnGenerator(dataDirectory, batchSize)
    return self.model.evaluate(generator)
    
  def TrainCompletion(self, dataDirectory, learningRateSchedule, batchSizeSchedule):
    '''Trains the shape completion component of the network.
    - Input dataDirectory: The directory containing the dataset (.mat files with g.t. completions.)
    - Input learningRateSchedule: A list of learning rates, one for each epoch.
    - Input batchSizeSchedule: A list of batch sizes with length len(learningRateSchedule).
    '''
    
    # freeze uncertainty layers
    for i in xrange(len(self.model.layers)):
      self.model.layers[i].trainable = self.model.layers[i].name[0:2] != "p-"
      
    # compile model
    self.model.compile(optimizer = self.model.optimizer, loss = self.ShapeCompletionLoss)
    
    # train
    for i in xrange(minimum(len(learningRateSchedule), len(batchSizeSchedule))):
      generator = PcnGenerator(dataDirectory, batchSizeSchedule[i], "Cpart", "Ccomp")
      keras.backend.set_value(self.model.optimizer.lr, learningRateSchedule[i])
      self.model.fit(generator, epochs = 1, shuffle = False)
      self.Save("pcn_model_backup.h5")
      
  def TrainCorrectnessProbability(self, dataDirectory, learningRateSchedule, batchSizeSchedule):
    '''Trains the correctness probability component of the network. Automatically
       computes Euclidean distance threshold that will balance the dataset.
    - Input dataDirectory: The directory containing the dataset (.mat files with g.t. completions.)
    - Input learningRateSchedule: A list of learning rates, one for each epoch.
    - Input batchSizeSchedule: A list of batch sizes with length len(learningRateSchedule).
    '''
    
    # freeze uncertainty layers
    for i in xrange(len(self.model.layers)):
      self.model.layers[i].trainable = self.model.layers[i].name[0:2] == "p-"
      
    # compile model
    self.model.compile(optimizer = self.model.optimizer, loss = self.CorrectnessProbabilityLoss)
    
    # train
    for i in xrange(minimum(len(learningRateSchedule), len(batchSizeSchedule))):
      generator = PcnGenerator(dataDirectory, batchSizeSchedule[i], "Cpart", "Ccomp")
      keras.backend.set_value(self.model.optimizer.lr, learningRateSchedule[i])
      self.model.fit(generator, epochs = 1, shuffle = False)
      self.Save("pcn_model_backup.h5")