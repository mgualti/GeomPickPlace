#!/usr/bin/env python
'''TODO'''

# python
import os
from time import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from numpy import array, concatenate, copy, ones, zeros
import matplotlib
# self
import point_cloud
from bonet.bonet_model import BonetModel
from place_blocks_sp_params import Parameters
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_blocks import PlannerBlocks
from geom_pick_place.planner_regrasp_sp import PlannerRegraspSP
from geom_pick_place.environment_blocks import EnvironmentBlocks

# uncomment when profiling
#import os; os.chdir("/home/mgualti/GeomPickPlace/Simulation")

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # system
  randomSeed = 0
  cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[0.30, 0.48], [-0.18, 0.18], [0.01, 1.00]]
  sceneViewWorkspace = [[0.00, 0.55], [-1.00, 1.00], [0.005, 1.00]]
  rowWorkspace = [[-0.48, -0.30], [-1.00, 1.00], [0.01, 1.00]]
  
  # task and environment
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/blocks_clouds_test"
  useGroundTruthSegmentation = False
  useGroundTruthCompletion = False
  useShapeCompletion = True
  nEpisodes = 200
  nObjects = 5
  
  # perception
  modelFileNameCompletion = "pcn_model_blocks.h5"
  modelFileNameSegmentation = "bonet_model_blocks.cptk"
  modelFileNameGraspPrediction = params["modelFileNameGraspPrediction"]
  modelFileNamePlacePrediction = params["modelFileNamePlacePrediction"]
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.005
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.80
  minPointsPerSegment = params["minPointsPerSegment"]
  
  # cost factors
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  insignificantCost = params["insignificantCost"]
  
  # task planning
  rowStart = point_cloud.WorkspaceCenter(rowWorkspace)
  rowStart[1] -= 0.18
  
  # regrasp planning
  temporaryPlacePosition = params["temporaryPlacePosition"]
  nGraspSamples = params["nGraspSamples"]
  halfAngleOfGraspFrictionCone = params["halfAngleOfGraspFrictionCone"]
  nTempPlaceSamples = params["nTempPlaceSamples"]
  nGoalPlaceSamples = params["nGoalPlaceSamples"]
  maxSearchDepth = params["maxSearchDepth"]
  maxIterations = params["maxIterations"]

  # visualization/saving
  showViewer = False
  showSteps = False
  showWarnings = False
  saveFileName = "results-blocks-sp.mat"
  params = locals()
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveFloatingHand()
  Tsensor[2, 3] += env.GetTableHeight()
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  rowWorkspace[2][0] += env.GetTableHeight()
  rowStart[2] = env.GetTableHeight()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
  # initialize model
  modelSegmentation = BonetModel(nInputPointsSegmentation, nObjects, deviceId,
    modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, deviceId, modelFileNameCompletion)
  
  # initialize planners
  taskPlanner = PlannerBlocks(env, rowStart, minPointsPerSegment)
  regraspPlanner = PlannerRegraspSP(env, temporaryPlacePosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples, modelFileNameGraspPrediction,
    modelFileNamePlacePrediction, taskCostFactor, regraspPlanStepCost, graspUncertCostFactor,
    placeUncertCostFactor, insignificantCost, maxSearchDepth, maxIterations, minPointsPerSegment)
  
  # RUN TEST =======================================================================================
  
  # start clock
  totalStartTime = time()
  
  # metrics being reccorded
  nPlaced = []; graspSuccess = []; graspAntipodal = []; tempPlaceStable = []; planLength = []; 
  orderCorrect = []; longestEndUp = []; taskPlanningTime = []; regraspPlanningTime = []
  
  for episode in xrange(randomSeed, nEpisodes):
    
    # set random seed
    seed(episode)
  
    # load objects into scene
    env.LoadInitialScene(nObjects, cloudDirectory, sceneWorkspace)
    
    '''if showSteps:
      raw_input("Loaded initial scene.")'''
      
    # for tracking the placed objects, as they become an obstacle at the goal
    placedObjects = zeros((0, 3))
      
    for t in xrange(nObjects):
      
      print("Episode {}.{} ====================================".format(episode, t))
      
      # move robot out of the way
      env.MoveRobotToHome()
      env.UnplotCloud()
      env.UnplotDescriptors()
      
      # get point cloud of the scene
      env.AddSensor()
      Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
      env.MoveSensorToPose(Tsensor)  
      cloud = env.GetCloud(sceneViewWorkspace)
      
      if cloud.shape[0] == 0:
        print("No points in cloud of scene!")
        env.RemoveObjectAtRandom()
        continue
      
      '''if showSteps:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene.")'''
        
      # get a point cloud of the block row
      Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(rowWorkspace)[0:2]
      env.MoveSensorToPose(Tsensor)
      blockRowCloud = env.GetCloud(rowWorkspace)
      
      '''if showSteps:
        env.PlotCloud(blockRowCloud)
        raw_input("Acquired cloud of placed objects.")'''
        
      # move sensor away so it doesn't cause collisions
      env.RemoveSensor()
        
      # segment objects
      startTime = time()
      segmentedClouds, segmentationProbs, _ = taskPlanner.SegmentObjects(cloud, modelSegmentation,
        scoreThreshSegmentation, useGroundTruthSegmentation)
      print("Segmentation took {} seconds.".format(time() - startTime))
      
      if showSteps:
        env.PlotCloud(concatenate(segmentedClouds),
          matplotlib.cm.viridis(concatenate(segmentationProbs)))
        raw_input("Showing {} segments.".format(len(segmentedClouds)))
        
      # complete objects
      startTime = time()
      if useShapeCompletion:
        completedClouds, completionProbs = taskPlanner.CompleteObjects(\
          segmentedClouds, modelCompletion, useGroundTruthCompletion)
      else:
        completedClouds = []; completionProbs = []
        for i in xrange(len(segmentedClouds)):
          completedClouds.append(copy(segmentedClouds[i]))
          completionProbs.append(ones(segmentedClouds[i].shape[0]))
      print("Completion took {} seconds.".format(time() - startTime))
      
      # compute normals for the target object
      normals = env.EstimateNormals(completedClouds)
      
      if showSteps:
        env.PlotCloud(concatenate(completedClouds),
          matplotlib.cm.viridis(concatenate(completionProbs)))
        raw_input("Showing {} completions".format(len(completedClouds)))
      
      # get goal poses from the task planner
      startTime = time()
      goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPoses(\
        completedClouds, normals, blockRowCloud)
      taskPlanningTime.append(time() - startTime)
      print("Task planner completed in {} seconds.".format(taskPlanningTime[-1]))
      
      if goalPoses is None:
        env.RemoveObjectNearestAnotherObject()
        continue
      
      # determine obstacles for regrasp planning
      obstaclesAtStart = []
      for targObjIdx in targObjIdxs:
        obstaclesAtStart.append(regraspPlanner.StackClouds(completedClouds, [targObjIdx]))
      obstaclesAtGoal = [concatenate([blockRowCloud, placedObjects])] * len(goalPoses)
      
      # get a sequence of grasp and place poses from the regrasp planner
      startTime = time()
      pickPlaces, plannedConfigs, goalIdx, targObjIdx = regraspPlanner.Plan(
        segmentedClouds, segmentationProbs, completedClouds, completionProbs, normals, goalPoses,
        goalCosts, targObjIdxs, obstaclesAtStart, obstaclesAtGoal)
      regraspPlanningTime.append(time() - startTime)
      print("Regrasp planner completed in {} seconds.".format(regraspPlanningTime[-1]))
      
      if len(pickPlaces) == 0:
        env.RemoveObjectNearestCloud(completedClouds[targObjIdx])
        continue
      
      planLength.append(len(pickPlaces))
      
      if showSteps:
        env.MoveRobotToHome()
        env.PlotDescriptors(pickPlaces)
        raw_input("Showing regrasp plan.")
      
      # actually do the pick places
      success, gSuccess, gAntipodal, tpStable  = env.ExecuteRegraspPlan( \
        pickPlaces, plannedConfigs, completedClouds[targObjIdx], showSteps)
      graspSuccess += gSuccess; graspAntipodal += gAntipodal; tempPlaceStable += tpStable
      
      # update the completed cloud of placed objects
      if success:
        placedObjects = regraspPlanner.StackClouds([placedObjects,
          point_cloud.Transform(goalPoses[goalIdx], completedClouds[targObjIdx])])
    
    # finished: evaluate and save packing result
    n, o, l = env.EvaluateArrangement()
    nPlaced.append(n); orderCorrect += o; longestEndUp += l
    print("Packed {} items with {} correctly ordered and {} longest end up.".format( \
      n, sum(o), sum(l)))
    
    if showSteps:
      env.MoveRobotToHome()
      env.UnplotDescriptors()
      raw_input("End of episode.")
    
    totalTime = time() - totalStartTime
    data = {"nPlaced":nPlaced, "graspSuccess":graspSuccess, "graspAntipodal":graspAntipodal,
      "tempPlaceStable":tempPlaceStable, "planLength":planLength, "orderCorrect":orderCorrect,
      "longestEndUp":longestEndUp, "taskPlanningTime":taskPlanningTime,
      "regraspPlanningTime":regraspPlanningTime, "totalTime":totalTime}
    data.update(params)
    savemat(saveFileName, data)

if __name__ == "__main__":
  main()