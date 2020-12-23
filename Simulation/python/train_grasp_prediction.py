#!/usr/bin/env python
'''TODO'''

# python
import os
# scipy
from numpy.random import  seed
# tensorflow
import tensorflow
# geom_pick_place
from geom_pick_place.pcn_model_sp import PcnModelSP

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  randomSeed = 0
  
  # completions
  nInputPoints = 128
  
  # training
  completionDirectory = "/home/mgualti/Data/GeomPickPlace/packing_grasps_train"
  modelFileName = "pcn_model_grasp_prediction_packing.h5"
  pcnArchitecture = True
  
  learningRateSchedule = [5.0e-4]*100 # for packing, PCN
  #learningRateSchedule = [1.0e-4]*50 # for bottles and blocks, PCN
  #learningRateSchedule = [1.0e-4]*100 # for packing, PointNetGPD
  batchSizeSchedule = [32]*100 # for packing, PCN
  #batchSizeSchedule = [32]*50 # for bottles and blocks, PCN
  #batchSizeSchedule = [32]*100 # for packing, PointNetGPD

  # visualization/saving
  showSteps = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  tensorflow.random.set_seed(randomSeed)
    
  # initialize model
  model = PcnModelSP(nInputPoints, deviceId, None, pcnArchitecture)
  
  if showSteps:
    print model.model.summary()
    raw_input("Initialized model.")

  # RUN TEST =======================================================================================
  
  model.Train(completionDirectory, learningRateSchedule, batchSizeSchedule)
  model.Save(modelFileName)

if __name__ == "__main__":
  main()