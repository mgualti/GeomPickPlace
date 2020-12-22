#!/usr/bin/env python
'''TODO'''

# python
import os
# scipy
from numpy.random import seed
# tensorflow
import tensorflow
# geom_pick_place
from bonet.bonet_model import BonetModel

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  deviceId = cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  randomSeed = 0
  
  # completion
  nInputPoints = 2048
  nCategories = 6
  
  # training
  segmentationDirectory = "/home/mgualti/Data/GeomPickPlace/packing_segmentations_train"
  modelFileName = "bonet_model_packing.cptk"
  exampleInputName = "cloud"
  exampleOutputName = "segmentation"
  batchSize = 4
  #learningRateSchedule = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
  #  0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.00025,
  #  0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025,
  #  0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.000125,
  #  0.000125, 0.000125, 0.000125, 0.000125, 0.000125, 0.000125, 0.000125, 0.000125, 0.000125,
  # 0.000125] # original schedule
  learningRateSchedule = [0.0005, 0.0005, 0.00025, 0.00025, 0.000125, 0.000125] # planned schedule

  # visualization/saving
  showSteps = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  tensorflow.random.set_seed(randomSeed)
    
  # initialize model
  model = BonetModel(nInputPoints, nCategories, deviceId)
  
  if showSteps:
    print model.model.summary()
    raw_input("Initialized model.")

  # RUN TEST =======================================================================================
  
  model.Train(segmentationDirectory, exampleInputName, exampleOutputName, learningRateSchedule, batchSize)
  model.Save(modelFileName)

if __name__ == "__main__":
  main()