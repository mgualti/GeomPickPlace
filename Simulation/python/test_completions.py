#!/usr/bin/env python
'''Shows completed point clouds for a trained PCN model.'''

# python
import os
from time import time
# scipy
from numpy.random import seed
# tensorflow
import tensorflow
# self
from geom_pick_place.pcn_model import PcnModel

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test1"
  deviceId = cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  randomSeed = 0
  
  # completions
  nInputPoints = 1024
  nOutputPoints = 1024
  errorThreshold = 0.006
  
  # testing
  batchSize = 32
  
  # objects
  completionDirectory = "/home/mgualti/Data/GeomPickPlace/packing_completions_" + scenario
  modelFileName = "pcn_model_packing.h5"

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  tensorflow.random.set_seed(randomSeed)
  
  # initialize model
  model = PcnModel(nInputPoints, nOutputPoints, errorThreshold, deviceId, modelFileName)

  # RUN TEST =======================================================================================
  
  startTime = time()
  averageLoss = model.TestCompletion(completionDirectory, batchSize)
  print("Testing completions took {} hours.".format((time() - startTime) / 3600))
  print("Average completions test loss: {}.".format(averageLoss))
  
  startTime = time()
  averageLoss = model.TestCorrectnessProbability(completionDirectory, batchSize)
  print("Testing correctness took {} hours.".format((time() - startTime) / 3600))
  print("Average correctness test loss: {}.".format(averageLoss))

if __name__ == "__main__":
  main()