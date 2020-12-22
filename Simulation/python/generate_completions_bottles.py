#!/usr/bin/env python
'''Generates dataset of complete and partial point clouds of individual bottles.'''

# python
import os
import shutil
import fnmatch
from time import time
# scipy
from scipy.io import savemat
from numpy.random import choice, seed
from numpy import arange, array, mean, repeat
# self
import point_cloud
from geom_pick_place.environment_bottles import EnvironmentBottles

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "train"
  randomSeed = 0 if scenario == "train" else 1
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [0.00, 1.00]]
  viewWorkspace = [[-1, 1], [-1, 1], [0.005, 1]]
  
  # objects
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_clouds_" + scenario
  completionDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_completions_" + scenario
  nObjectPoses = 200 if scenario == "train" else 20
  
  # completion
  nInputPoints = 1024
  nOutputPoints = 1024

  # visualization/saving
  showViewer = False
  showSteps = False
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
  env.RemoveRobot()
  #env.RemoveTable()
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
  sceneWorkspace[2][0] += env.GetTableHeight()
  viewWorkspace[2][0] += env.GetTableHeight()
  
  # remove files in completion directory
  if os.path.exists(completionDirectory):
    response = raw_input("Overwrite existing directory? (Y/N): ")
    if response.lower() != "y": return
    shutil.rmtree(completionDirectory)
  os.mkdir(completionDirectory)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  cloudFileNames = os.listdir(cloudDirectory)
  cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
  cloudFileNames.sort(key = lambda fileName: int(fileName[:-4]))
  
  startTime = time()
  completeCloudSizes = []
  partialCloudSizes = []
  databaseSize = 0
  
  for i, cloudFileName in enumerate(cloudFileNames):
    
    # load the next object
    obj = env.LoadObjectFromFullCloudFile(cloudDirectory, cloudFileName, "object-{}".format(i))
    
    for j in xrange(nObjectPoses):
      
      print("Object {}, pose {}.".format(i, j))
      
      # randomly perterb the object's orientation
      Tobject = env.RandomizeObjectPose(obj, sceneWorkspace)
      if showSteps: raw_input("Placed object.")
      
      # acquire top-view point cloud of the object
      Cpart = env.GetCloud(viewWorkspace)
      Ccomp = obj.cloud
      if showSteps:
        env.PlotCloud(Cpart)
        raw_input("Acquired partial cloud.")
        
      partialCloudSizes.append(Cpart.shape[0])      
      completeCloudSizes.append(Ccomp.shape[0])
        
      # sample partial and complete clouds to fixed sizes
      idx = choice(arange(Cpart.shape[0]), size = nInputPoints)
      Cpart = Cpart[idx, :]
      idx = choice(arange(Ccomp.shape[0]), size = nOutputPoints)
      Ccomp = Ccomp[idx, :]
        
      # shift clouds (don't shift in z direction as this will remove height information)
      CpartCenter = array([mean(Cpart, axis = 0)]); CpartCenter[0, 2] = 0
      Cpart -= repeat(CpartCenter, Cpart.shape[0], axis = 0)
      Ccomp = point_cloud.Transform(Tobject, Ccomp)
      Ccomp -= repeat(CpartCenter, Ccomp.shape[0], axis = 0)
      
      if showSteps:
        env.PlotCloud(Cpart)
        raw_input("Shifted partial cloud.")
        env.PlotCloud(Ccomp)
        raw_input("Shifted complete cloud.")
      
      # save partial and ground truth clouds
      Cpart = Cpart.astype('float32')
      Ccomp = Ccomp.astype('float32')
      databaseSize += 4 * (Cpart.size + Ccomp.size)
      data = {"Cpart":Cpart, "Ccomp":Ccomp}
      fileName = completionDirectory + "/" + "object-" + str(i) + "-scene-" + str(j) + ".mat"
      savemat(fileName, data)
    
    env.RemoveObjectSet([obj])
  
  print("Average partial/complete cloud size: {} / {}".format(mean(partialCloudSizes),
    mean(completeCloudSizes)))
  print("Database size: {} MB.".format(databaseSize / 1024**2))
  print("Completed in {} hours.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()