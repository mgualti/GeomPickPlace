#!/usr/bin/env python
'''Shows segmented point clouds for a trained BoNet model.'''

# python
from time import time
# scipy
import matplotlib
from numpy.random import normal, seed
from numpy import array, pi
# self
from bonet.bonet_model import BonetModel
from geom_pick_place.planner import Planner
from geom_pick_place.environment_bottles import EnvironmentBottles

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "train"
  addSensorNoise = True
  nEpisodes = 100
  randomSeed = 8
  
  # robot
  viewKeepout = 0.70
  viewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  
  # environment
  cloudDirectoryBottles = "/home/mgualti/Data/GeomPickPlace/bottles_clouds_" + scenario
  cloudDirectoryCoasters = "/home/mgualti/Data/GeomPickPlace/coasters_clouds_" + scenario
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [-1.00, 1.00]]
  nObjects = 2
  nCategories = 2 * nObjects
  
  # segmentation
  scoreThresh = 0.00
  nInputPoints = 2048
  modelFileName = "bonet_model_bottles.cptk"
  
  # visualization/saving
  showViewer = True
  showSteps = True
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
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
  model = BonetModel(nInputPoints, nCategories, -1, modelFileName)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  for episode in xrange(randomSeed, nEpisodes):
    
    # set random seed
    seed(episode)
  
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
      cloud = env.GetCloud(viewWorkspace)
      
      if showSteps:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene with {} points.".format(cloud.shape[0]))
        
      # add sensor noise
      if addSensorNoise:
        cloud = env.AddSensorNoise(cloud, 1.5 * (pi / 180), 4, 0.0125, 0.002, 0.001)
        if showSteps:
          env.PlotCloud(cloud)
          raw_input("Added sensor noise.")
      
      # segment objects
      startTime = time()
      segmentedClouds, segmentProbs = taskPlanner.SegmentObjects(\
        cloud, model, sceneWorkspace, scoreThresh, False)
      print("Segmentation took {} seconds.".format(time() - startTime))
      
      if showSteps:
        for i, c in enumerate(segmentedClouds):
          env.PlotCloud(c, matplotlib.cm.viridis(segmentProbs[i]))
          raw_input("Segment {}/{} with uncertainty in [{}, {}].".format(\
            i+1, len(segmentedClouds), min(segmentProbs[i]), max(segmentProbs[i])))
      
      # simulate placing an object
      if episode % 2 == 0:
        # remove an object
        env.RemoveBottleAtRandom()
      else:  
        # place an object
        env.PlaceBottleOnCoasterAtRandom()

if __name__ == "__main__":
  main()