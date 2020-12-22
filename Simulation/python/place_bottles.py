#!/usr/bin/env python
'''TODO'''

# python
import os
from time import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from numpy import array, concatenate, copy, ones
import matplotlib
# self
import point_cloud
from place_bottles_params import Parameters
from bonet.bonet_model import BonetModel
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_bottles import PlannerBottles
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_bottles import EnvironmentBottles

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
  
  # task and environment
  cloudDirectoryBottles = "/home/mgualti/Data/GeomPickPlace/bottles_clouds_test"
  cloudDirectoryCoasters = "/home/mgualti/Data/GeomPickPlace/coasters_clouds_test"
  useGroundTruthSegmentation = False
  useGroundTruthCompletion = False
  useShapeCompletion = True
  nEpisodes = 500
  nObjects = 2
  
  # perception
  modelFileNameCompletion = "pcn_model_bottles.h5"
  modelFileNameSegmentation = "bonet_model_bottles.cptk"
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.007
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.75
  minPointsPerSegment = params["minPointsPerSegment"]
  
  # cost factors
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  segmUncertCostFactor = params["segmUncertCostFactor"]
  compUncertCostFactor = params["compUncertCostFactor"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  antipodalCostFactor = params["antipodalCostFactor"]
  insignificantCost = params["insignificantCost"]
  
  # task planning
  nGoalOrientations = 30
  
  # regrasp planning
  temporaryPlacePosition = params["temporaryPlacePosition"]
  nGraspSamples = params["nGraspSamples"]
  halfAngleOfGraspFrictionCone = params["halfAngleOfGraspFrictionCone"]
  nTempPlaceSamples = params["nTempPlaceSamples"]
  nGoalPlaceSamples = params["nGoalPlaceSamples"]
  maxSearchDepth = params["maxSearchDepth"]
  maxIterations = params["maxIterations"]

  # visualization/saving
  showViewer = True
  showSteps = True
  showWarnings = False
  saveFileName = "results-bottles-{}.mat".format(randomSeed)
  params = locals()
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
  env.RemoveFloatingHand()
  Tsensor[2, 3] += env.GetTableHeight()
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
  # initialize model
  modelSegmentation = BonetModel(
    nInputPointsSegmentation, 2 * nObjects, deviceId, modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, deviceId, modelFileNameCompletion)
  
  # initialize planners
  taskPlanner = PlannerBottles(env, nGoalOrientations, minPointsPerSegment)
  regraspPlanner = PlannerRegrasp(env, temporaryPlacePosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor,
    regraspPlanStepCost, segmUncertCostFactor, compUncertCostFactor, graspUncertCostFactor,
    placeUncertCostFactor, antipodalCostFactor, insignificantCost, maxSearchDepth, maxIterations,
    minPointsPerSegment)
  
  # RUN TEST =======================================================================================
  
  # start clock
  totalStartTime = time()
  
  # metrics being reccorded
  nPlaced = []; graspSuccess = []; graspAntipodal = []; tempPlaceStable = []; planLength = [];
  taskPlanningTime = []; regraspPlanningTime = []
  
  for episode in xrange(randomSeed, nEpisodes):
    
    # set random seed
    seed(episode)
  
    # load objects into scene
    env.LoadInitialScene(nObjects, cloudDirectoryBottles, cloudDirectoryCoasters, sceneWorkspace)
    
    '''if showSteps:
      raw_input("Loaded initial scene.")'''
      
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
        env.RemoveBottleAtRandom()
        continue
      
      if showSteps:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene.")
        
      # move sensor away so it doesn't cause collisions
      env.RemoveSensor()
        
      # segment objects
      startTime = time()
      freeBottles, freeBottleProbs, placedBottles, freeCoasters, occupiedCoasters, _ = \
        taskPlanner.SegmentObjects(cloud, modelSegmentation, scoreThreshSegmentation,
        useGroundTruthSegmentation, nObjects)
      print("Segmentation took {} seconds.".format(time() - startTime))
      
      if len(freeBottles) == 0:
        print("No free bottles detected!")
        env.RemoveBottleAtRandom()
        continue
      
      if showSteps:
        env.PlotCloud(concatenate(freeBottles),
          matplotlib.cm.viridis(concatenate(freeBottleProbs)))
        raw_input("Showing {} free bottles.".format(len(freeBottles)))
        if len(placedBottles) > 0:
          env.PlotCloud(concatenate(placedBottles))
          raw_input("Showing {} placed bottles.".format(len(placedBottles)))
        if len(freeCoasters) > 0:
          env.PlotCloud(concatenate(freeCoasters))
          raw_input("Showing {} free coasters.".format(len(freeCoasters)))
        if len(occupiedCoasters) > 0:
          env.PlotCloud(concatenate(occupiedCoasters))
          raw_input("Showing {} occupied coasters.".format(len(occupiedCoasters)))
        
      # complete objects
      startTime = time()
      if useShapeCompletion:
        completedFreeBottles, completionProbs = taskPlanner.CompleteObjects(\
          freeBottles, modelCompletion, useGroundTruthCompletion)
      else:
        completedFreeBottles = []; completionProbs = []
        for i in xrange(len(freeBottles)):
          completedFreeBottles.append(copy(freeBottles[i]))
          completionProbs.append(ones(freeBottles[i].shape[0]))
      print("Completion took {} seconds.".format(time() - startTime))
      
      # compute normals for the target object
      normals = env.EstimateNormals(completedFreeBottles)
      
      if showSteps:
        env.PlotCloud(concatenate(completedFreeBottles),
          matplotlib.cm.viridis(concatenate(completionProbs)))
        raw_input("Showing {} completions".format(len(completedFreeBottles)))
      
      # get goal poses from the task planner
      startTime = time()
      goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPoses(completedFreeBottles, freeCoasters)
      taskPlanningTime.append(time() - startTime)
      print("Task planner completed in {} seconds.".format(taskPlanningTime[-1]))
      
      if len(goalPoses) == 0:
        env.RemoveBottleNearestAnotherObject()
        continue
      
      # determine obstacles for regrasp planning
      obstacleClouds = []
      for targObjIdx in targObjIdxs:
        obstacleClouds.append(regraspPlanner.StackClouds(\
          completedFreeBottles + placedBottles + freeCoasters + occupiedCoasters, [targObjIdx]))
      
      # get a sequence of grasp and place poses from the regrasp planner
      startTime = time()
      pickPlaces, plannedConfigs, goalIdx, targObjIdx = regraspPlanner.Plan(
        freeBottles, freeBottleProbs, completedFreeBottles, completionProbs, normals, goalPoses,
        goalCosts, targObjIdxs, obstacleClouds, obstacleClouds)
      regraspPlanningTime.append(time() - startTime)
      print("Regrasp planner completed in {} seconds.".format(regraspPlanningTime[-1]))
      
      if len(pickPlaces) == 0:
        env.RemoveBottleNearestCloud(completedFreeBottles[targObjIdx])
        continue
      
      planLength.append(len(pickPlaces))
      
      if showSteps:
        env.MoveRobotToHome()
        env.PlotDescriptors(pickPlaces)
        raw_input("Showing regrasp plan.")
      
      # actually do the pick places
      gSuccess, gAntipodal, tpStable  = env.ExecuteRegraspPlan( \
        pickPlaces, plannedConfigs, completedFreeBottles[targObjIdx], showSteps)
      graspSuccess += gSuccess; graspAntipodal += gAntipodal; tempPlaceStable += tpStable
    
    # finished: evaluate arrangement and save result
    nPlaced.append(env.EvaluateArrangement())
    print("Correctly placed {} bottles.".format(nPlaced[-1]))
    
    if showSteps:
      env.MoveRobotToHome()
      env.UnplotDescriptors()
      raw_input("End of episode.")
    
    totalTime = time() - totalStartTime
    data = {"nPlaced":nPlaced, "graspSuccess":graspSuccess, "graspAntipodal":graspAntipodal,
      "tempPlaceStable":tempPlaceStable, "planLength":planLength,
      "taskPlanningTime":taskPlanningTime, "regraspPlanningTime":regraspPlanningTime,
      "totalTime":totalTime}
    data.update(params)
    savemat(saveFileName, data)

if __name__ == "__main__":
  main()