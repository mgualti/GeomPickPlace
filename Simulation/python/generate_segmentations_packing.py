#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
from time import time
# scipy
from scipy.io import savemat
from numpy.random import choice, seed
from numpy import arange, array
# self
import point_cloud
from geom_pick_place.environment_packing import EnvironmentPacking

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "train"
  
  if scenario == "train":
    randomSeed = 0
  elif scenario == "test1":
    randomSeed = 1
  elif scenario == "test2":
    randomSeed = 2
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [0.0, 1.00]]
  sceneViewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  
  # objects
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/packing_clouds_" + scenario
  segmentationDirectory = "/home/mgualti/Data/GeomPickPlace/packing_segmentations_" + scenario
  nEpisodes = 166667 if scenario == "train" else 5000
  nObjects = 6
  
  # segmentation
  nInputPoints = 2048

  # visualization/saving
  showViewer = False
  showSteps = False
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveBox()
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  
  # remove files in segmentation directory
  if os.path.exists(segmentationDirectory):
    response = raw_input("Overwrite {}? (Y/N): ".format(segmentationDirectory))
    if response.lower() != "y": return
    shutil.rmtree(segmentationDirectory)
  os.mkdir(segmentationDirectory)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  databaseSize = 0
  startTime = time()
  
  for episode in xrange(nEpisodes):
  
    # load objects into scene
    env.LoadInitialScene(nObjects, cloudDirectory, sceneWorkspace)
    
    if showSteps:
      raw_input("Loaded initial scene.")
      
    for t in xrange(nObjects):
      
      print("Episode {}.{} ====================================".format(episode, t))   
      
      # get point cloud of the scene
      env.MoveSensorToPose(Tsensor)
      cloud = env.GetCloud(sceneViewWorkspace)
      
      if showSteps:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene with {} points.".format(cloud.shape[0]))
        
      if cloud.shape[0] == 0:
        print("Cloud has no points!")
        env.RemoveObjectAtRandom()
        continue
      
      # sample cloud to fixed input size
      idx = choice(arange(cloud.shape[0]), size = nInputPoints)
      cloud = cloud[idx, :]
      
      if showSteps:
        env.PlotCloud(cloud)
        raw_input("Downsampled cloud to {} points.".format(cloud.shape[0]))
        
      # determine scene segmentation
      segmentation = env.GetSegmentation(cloud, nObjects)
      
      if showSteps:
        for i in xrange(nObjects):
          if sum(segmentation[:, i]) == 0: continue
          env.PlotCloud(cloud[segmentation[:, i], :])
          raw_input("Segmentation-{}".format(i))
      
      # save segmentation example
      cloud = cloud.astype('float32')
      segmentation = segmentation.astype('float32')
      databaseSize += 4 * (cloud.size + segmentation.size)
      data = {"cloud":cloud, "segmentation":segmentation}
      fileName = segmentationDirectory + "/" + "scene-" + str(episode) + "-time-" + str(t) + ".mat"
      savemat(fileName, data)
      
      # simulate packign an object
      env.RemoveObjectAtRandom()
      
  print("Database size: {} MB.".format(databaseSize / 1024**2))
  print("Completed in {} hours.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()