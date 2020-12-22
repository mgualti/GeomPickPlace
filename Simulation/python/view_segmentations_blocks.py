#!/usr/bin/env python
'''Shows completed point clouds for a trained PCN model.'''

# python
from time import time
# scipy
import matplotlib
from numpy.random import seed
from numpy import array, ones
# self
from bonet.bonet_model import BonetModel
from geom_pick_place.planner import Planner
from geom_pick_place.environment_blocks import EnvironmentBlocks

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test"
  nEpisodes = 100
  randomSeed = 0
  
  # robot
  viewKeepout = 0.70
  viewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  
  # environment
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/blocks_clouds_" + scenario
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [-1.00, 1.00]]
  nObjects = 5
  
  # segmentation
  scoreThresh = 0.70
  nInputPoints = 2048
  modelFileName = "bonet_model_blocks.cptk"
  
  # visualization/saving
  showViewer = True
  showSteps = True
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveTable()
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  sceneWorkspace[2][0] += env.GetTableHeight()
  viewWorkspace[2][0] += env.GetTableHeight()
  
  # initialize task planner
  taskPlanner = Planner(env)
  
  # initialize model
  model = BonetModel(nInputPoints, nObjects, -1, modelFileName)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  for episode in xrange(randomSeed, nEpisodes):
    
    # set random seed
    seed(episode)
  
    # load objects into scene
    env.LoadInitialScene(nObjects, cloudDirectory, sceneWorkspace)
    
    if showSteps:
      raw_input("Loaded initial scene.")
      
    for t in xrange(nObjects):
      
      print("Episode {}.{} ====================================".format(episode, t))   
      
      # get point cloud of the scene
      env.MoveSensorToPose(Tsensor)
      cloud = env.GetCloud(viewWorkspace)
      
      if showSteps:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene with {} points.".format(cloud.shape[0]))
      
      # segment objects
      startTime = time()
      segmentedClouds, segmentCosts = taskPlanner.SegmentObjects(\
        cloud, model, sceneWorkspace, scoreThresh, False)
      print("Segmentation took {} seconds.".format(time() - startTime))
      
      if showSteps:
        for i, c in enumerate(segmentedClouds):
          colors = matplotlib.cm.viridis(10 * segmentCosts[i] * ones(c.shape[0]))
          env.PlotCloud(c, colors)
          raw_input("Segment {}/{} with uncertainty {}.".format(\
            i+1, len(segmentedClouds), segmentCosts[i]))
      
      # simulate packign an object
      env.RemoveObjectAtRandom()

if __name__ == "__main__":
  main()