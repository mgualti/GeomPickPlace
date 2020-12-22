'''Loads point cloud completion dataset from files in batches.'''

# python
import os
import fnmatch
# scipy
from scipy.io import loadmat
from numpy.random import shuffle
from numpy import arange, stack
# tensorflow
from tensorflow import keras
# self

class PcnGenerator(keras.utils.Sequence):
  
  def __init__(self, dataDirectory, batchSize, inputName, labelName):
    '''Returns a keras Sequence object which can be used by fit_generator.
    - Input dataDirectory: The directory from which to load point cloud completion examples.
    - Input batchSize: The size of the batches to use during training.
    - Input inputName: TODO
    - Input labelName: TODO
    - Returns a PcnGenerator instance.
    '''
    
    keras.utils.Sequence.__init__(self)
    
    self.dataDirectory = dataDirectory
    self.batchSize = batchSize
    self.inputName = inputName
    self.labelName = labelName
    
    self.fileNames = os.listdir(dataDirectory)
    self.fileNames = fnmatch.filter(self.fileNames, "*.mat")
    
    self.indices = arange(len(self.fileNames))
    self.on_epoch_end()
    
  def __getitem__(self, batchIndex):
    '''Generate 1 batch of data.'''
    
    X = []; Y = []
    for i in xrange(self.batchSize):
      fileIdx = self.indices[batchIndex * self.batchSize + i]
      data = loadmat(self.dataDirectory + "/" + self.fileNames[fileIdx])
      X.append(data[self.inputName]); Y.append(data[self.labelName])
        
    return stack(X, axis = 0), stack(Y, axis = 0)
    
  def __len__(self):
    '''Number of batches per epoch.'''
    
    return int(len(self.fileNames) / self.batchSize)
    
  def on_epoch_end(self):
    '''Called at the end of each epoch. Automatically shuffles data.'''
    
    shuffle(self.indices)