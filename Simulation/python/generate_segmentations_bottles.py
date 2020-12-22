#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
from time import time
# scipy
from scipy.io import savemat
from numpy import arange, array, pi
from numpy.random import choice, normal, seed
# self
import point_cloud
from geom_pick_place.environment_bottles import EnvironmentBottles

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "train"
  randomSeed = 0 if scenario == "train" else 1
  
  addSensorNoise = True
  noiseString = "noise_" if addSensorNoise else ""
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [0.0, 1.00]]
  sceneViewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  
  # objects
  cloudDirectoryBottles = "/home/mgualti/Data/GeomPickPlace/bottles_clouds_" + scenario
  cloudDirectoryCoasters = "/home/mgualti/Data/GeomPickPlace/coasters_clouds_" + scenario
  segmentationDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_segmentations_" + \
    noiseString + scenario
  nEpisodes = 250000 if scenario == "train" else 20000
  nObjects = 2
  
  # segmentation
  nInputPoints = 2048
  nCategories = 2 * nObjects

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
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  
  # remove files in segmentation directory
  if os.path.exists(segmentationDirectory):
    response = raw_input("Overwrite existing directory? (Y/N): ")
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
    env.LoadInitialScene(nObjects, cloudDirectoryBottles, cloudDirectoryCoasters, sceneWorkspace)
    
    if showSteps:
      raw_input("Loaded initial scene.")
      
    for t in xrange(nObjects):
      
      print("Episode {}.{} ====================================".format(episode, t))   
      
      # get point cloud of the scene
      if addSensorNoise:
        targetP = normal(loc = [0, 0, 0], scale = [0.03, 0.03, 0.00], size = 3)
        T = env.GetSensorPoseGivenUp(Tsensor[0:3, 3], targetP, [1, 0, 0])
        env.MoveSensorToPose(T)    
      cloud = env.GetCloud(sceneViewWorkspace)
      
      if showSteps:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene with {} points.".format(cloud.shape[0]))
        
      # add sensor noise
      if addSensorNoise:
        cloud = env.AddSensorNoise(cloud, 1.5 * (pi / 180), 4, 0.0125, 0.002, 0.001)
        if showSteps:
          env.PlotCloud(cloud)
          raw_input("Added sensor noise.")
          
      if cloud.shape[0] == 0:
        print("Cloud has no points!")
        env.RemoveBottleAtRandom()
        continue
      
      # sample cloud to fixed input size
      idx = choice(arange(cloud.shape[0]), size = nInputPoints)
      cloud = cloud[idx, :]
      
      '''if showSteps:
        env.PlotCloud(cloud)
        raw_input("Downsampled cloud to {} points.".format(cloud.shape[0]))'''
        
      # determine scene segmentation
      segmentation = env.GetSegmentation(cloud, nCategories)
      
      if showSteps:
        for i in xrange(nCategories):
          if sum(segmentation[:, i]) == 0: continue
          env.PlotCloud(cloud[segmentation[:, i], :])
          raw_input("Segmentation-{}".format(i))
      
      # save segmentation example
      segmentation = segmentation.astype('float32')
      cloud = cloud.astype('float32')
      databaseSize += 4 * (cloud.size + segmentation.size)
      data = {"cloud":cloud, "segmentation":segmentation}
      fileName = segmentationDirectory + "/" + "scene-" + str(episode) + "-time-" + str(t) + ".mat"
      savemat(fileName, data)
      
      # simulate placing an object
      if episode % 2 == 0:
        # remove an object
        env.RemoveBottleAtRandom()
      else:  
        # place an object
        env.PlaceBottleOnCoasterAtRandom()
      
  print("Database size: {} MB.".format(databaseSize / 1024**2))
  print("Completed in {} hours.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()