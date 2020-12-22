#!/usr/bin/env python
'''TODO'''

# python
import os
import sys
import fnmatch
from time import time
# scipy
from scipy.io import loadmat
from numpy.random import seed
from numpy import array, mean, squeeze
# tensorflow
import tensorflow
# self
from geom_pick_place.pcn_model_sp import PcnModelSP

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test1"
  deviceId = cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  randomSeed = 0
  
  # completions
  nInputPoints = 128
  
  # testing
  batchSize = 32
  precisionThreshold = 0.95
  
  # objects
  graspDirectory = "/home/mgualti/Data/GeomPickPlace/packing_places_" + scenario
  modelFileName = "pcn_model_place_prediction_packing.h5"

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  tensorflow.random.set_seed(randomSeed)
  
  # initialize model
  model = PcnModelSP(nInputPoints, deviceId, modelFileName)

  # RUN TEST =======================================================================================
  
  startTime = time()
  averageLoss, accuracy = model.Test(graspDirectory, batchSize)
  precision = model.TestPrecision(graspDirectory, batchSize, precisionThreshold)
  print("Testing place prediction took {} hours.".format((time() - startTime) / 3600))
  print("Average test loss: {}.".format(averageLoss))
  print("Test accuracy: {}.".format(accuracy))
  print("Test precision at threshold {}: {}".format(precisionThreshold, precision))

if __name__ == "__main__":
  main()