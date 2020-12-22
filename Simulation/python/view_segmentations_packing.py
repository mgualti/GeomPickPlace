#!/usr/bin/env python
'''Shows segmented point clouds for a trained BoNet model.'''

# python
from time import time
# scipy
import matplotlib
from numpy import array
from numpy.random import seed
# self
from bonet.bonet_model import BonetModel
from geom_pick_place.planner_packing import PlannerPacking
from geom_pick_place.environment_packing import EnvironmentPacking

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test1"
  nEpisodes = 100
  randomSeed = 0
  
  # robot
  viewKeepout = 0.60
  viewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  
  # environment
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/packing_clouds_" + scenario
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [-1.00, 1.00]]
  nObjects = 6
  
  # segmentation
  scoreThresh = 0.00
  nInputPoints = 2048
  minPointsPerSegment = 1
  modelFileName = "bonet_model_packing.cptk"
  
  # visualization/saving
  showViewer = True
  showSteps = True
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveBox()
  env.RemoveTable()
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  sceneWorkspace[2][0] += env.GetTableHeight()
  viewWorkspace[2][0] += env.GetTableHeight()
  
  # initialize task planner
  taskPlanner = PlannerPacking(env, 0, 0, 0, 0, 0, 0, minPointsPerSegment)
  
  # initialize model
  model = BonetModel(nInputPoints, nObjects, -1, modelFileName)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  for episode in xrange(nEpisodes):
  
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
      segmentedClouds, segmentationProbs, _ = taskPlanner.SegmentObjects(
        cloud, model, scoreThresh, False)
      print("Segmentation took {} seconds.".format(time() - startTime))
      
      if showSteps:
        for i, c in enumerate(segmentedClouds):
          env.PlotCloud(c, matplotlib.cm.viridis(segmentationProbs[i]))
          raw_input("Segment {}/{} with uncertainty in [{}, {}].".format(\
            i+1, len(segmentedClouds), min(segmentationProbs[i]), max(segmentationProbs[i])))
      
      # simulate packign an object
      env.RemoveObjectAtRandom()

if __name__ == "__main__":
  main()