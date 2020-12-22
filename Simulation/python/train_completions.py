#!/usr/bin/env python
'''TODO'''

# python
import os
# scipy
from numpy.random import  seed
# tensorflow
import tensorflow
# geom_pick_place
from geom_pick_place.pcn_model import PcnModel

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  randomSeed = 0
  
  # completions
  nInputPoints = 1024
  nOutputPoints = 1024
  errorThreshold = 0.006
  
  # training
  completionDirectory = "/home/mgualti/Data/GeomPickPlace/packing_completions_train"
  modelFileName = "pcn_model_packing.h5"
  
  completionLearningRateSchedule = [2.0e-4, 1.0e-4, 0.5e-4, 0.25e-4, 0.1e-4]
  correctnessLearningRateSchedule = [4.0e-3, 3.0e-3, 2.0e-3, 1.0e-3, 0.5e-3]
  batchSizeSchedule = [32, 32, 32, 32, 32]

  # visualization/saving
  showSteps = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  tensorflow.random.set_seed(randomSeed)
    
  # initialize model
  model = PcnModel(nInputPoints, nOutputPoints, errorThreshold, deviceId)
  
  if showSteps:
    print model.model.summary()
    raw_input("Initialized model.")

  # RUN TEST =======================================================================================
  
  print("Training completion.")
  model.TrainCompletion( \
    completionDirectory, completionLearningRateSchedule, batchSizeSchedule)
  print("Training correctness probability with threshold {} mm.".format(errorThreshold * 1000))
  model.TrainCorrectnessProbability( \
    completionDirectory, correctnessLearningRateSchedule, batchSizeSchedule)
  model.Save(modelFileName)

if __name__ == "__main__":
  main()