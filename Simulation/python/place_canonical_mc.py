#!/usr/bin/env python
'''TODO'''

# python
import os
from time import time
# scipy
from scipy.io import savemat
from numpy import array, concatenate, copy, ones, zeros
import matplotlib
# self
import point_cloud
from place_canonical_mc_params import Parameters
from bonet.bonet_model import BonetModel
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_regrasp_mc import PlannerRegraspMC
from geom_pick_place.planner_canonical import PlannerCanonical
from geom_pick_place.environment_canonical import EnvironmentCanonical

# uncomment when profiling
#import os; os.chdir("/home/mgualti/GeomPickPlace/Simulation")

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # system
  startEpisode = 0
  cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[0.30, 0.48], [-0.18, 0.18], [0.01, 1.00]]
  sceneViewWorkspace = [[0.00, 0.55], [-1.00, 1.00], [0.005, 1.00]]
  goalPlacementPosition = (-0.39, 0.00)
  
  # task and environment
  sceneDirectory = "/home/mgualti/Data/GeomPickPlace/canonical_scenes_test1_1objects"
  useGroundTruthSegmentation = True
  useGroundTruthCompletion = False
  useShapeCompletion = True
  nEpisodes = 2000
  nObjects = 6
  
  # perception
  modelFileNameCompletion = "pcn_model_packing.h5"
  modelFileNameSegmentation = "bonet_model_packing.cptk"
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.006
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.60
  nCompletionSamples = 50
  certainEquivSegmProb = 0.75
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
  maxGoalsPerWorker = params["maxGoalsPerWorker"]
  
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
  saveFileName = "results-canonical-mc.mat"
  params = locals()
  
  # INITIALIZATION ================================================================================= 
  
  # initialize environment
  env = EnvironmentCanonical(showViewer, showWarnings)
  env.RemoveFloatingHand()
  Tsensor[2, 3] += env.GetTableHeight()
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
  # initialize model
  modelSegmentation = BonetModel(
    nInputPointsSegmentation, nObjects, deviceId, modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, deviceId, modelFileNameCompletion)
  
  # initialize planners
  taskPlanner = PlannerCanonical(env, maxGoalsPerWorker, minPointsPerSegment)
  regraspPlanner = PlannerRegraspMC(env, temporaryPlacePosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor,
    regraspPlanStepCost, segmUncertCostFactor, compUncertCostFactor, graspUncertCostFactor,
    placeUncertCostFactor, antipodalCostFactor, insignificantCost, maxSearchDepth, maxIterations,
    minPointsPerSegment)
  
  # RUN TEST =======================================================================================
  
  # start clock
  totalStartTime = time()  
  
  # metrics being recorded
  nPlaced = []; graspSuccess = []; graspAntipodal = []; tempPlaceStable = []; planLength = [];
  taskPlanningTime = []; regraspPlanningTime = []
  
  for episode in xrange(startEpisode, nEpisodes):
  
    # load objects into scene
    env.LoadInitialScene(sceneDirectory, episode)
    
    '''if showSteps:
      raw_input("Loaded initial scene.")'''
    
    success = False
    for t in xrange(1):
      
      print("Episode {}.{}. ====================================".format(episode, t))
      
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
        env.RemoveObjectNearestAnotherObject()
        continue
      
      if showSteps and cloud.shape[0] > 0:
        env.PlotCloud(cloud)
        raw_input("Acquired cloud of scene.")
        
      # move sensor away so it doesn't cause collisions
      env.RemoveSensor()
        
      # segment objects
      startTime = time()
      segmentedClouds, segmentationProbs, segmentationDist = taskPlanner.SegmentObjects(
        cloud, modelSegmentation,scoreThreshSegmentation, useGroundTruthSegmentation)
      print("Segmentation took {} seconds.".format(time() - startTime))
      
      if len(segmentedClouds) == 0:
          print("No segments found.")
          env.RemoveObjectNearestAnotherObject()
          continue
      
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
      
      if showSteps:
        env.PlotCloud(concatenate(completedClouds),
          matplotlib.cm.viridis(concatenate(completionProbs)))
        raw_input("Showing {} completions".format(len(completedClouds)))
          
      # compute normals for the target object
      normals = env.EstimateNormals(completedClouds)
      
      # sample shape completions
      startTime = time()
      completionSamples, completionSampleNormals = regraspPlanner.SampleShapeCompletions(cloud,
        segmentedClouds, segmentationDist, modelCompletion, completedClouds, completionProbs,
        nCompletionSamples, certainEquivSegmProb)
      print("Sampling shape completions took {} seconds.".format(time() - startTime))
        
      if showSteps:
        for i in xrange(nCompletionSamples):
          sample = [completionSamples[j][i] for j in xrange(len(segmentedClouds))]
          objectId = [(1 + j) * ones(nOutputPointsCompletion) / len(segmentedClouds) for \
            j in xrange(len(segmentedClouds))]
          env.PlotCloud(concatenate(sample), matplotlib.cm.viridis(concatenate(objectId)))
          raw_input("Showing completion sample {}.".format(i))
          
      # get goal poses from the high level planner
      startTime = time()
      goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPoses(
        completedClouds, goalPlacementPosition)
      taskPlanningTime.append(time() - startTime)
      print("Task planner completed in {} seconds.".format(taskPlanningTime[-1]))
      
      if len(goalPoses) == 0:
        print("No goal poses found.")
        env.RemoveObjectNearestAnotherObject()
        continue
        
      # determine obstacles for regrasp planning
      obstacleCloudsAtStart = []; obstacleCloudsAtGoal = []
      for targObjIdx in targObjIdxs:
        obstacleCloudsAtStart.append(regraspPlanner.StackClouds(completedClouds, [targObjIdx]))
        obstacleCloudsAtGoal.append(zeros((0, 3)))
        
      # get a sequence of grasp and place poses from the regrasp planner
      startTime = time()
      pickPlaces, plannedConfigs, goalIdx, targObjIdx = regraspPlanner.Plan(completedClouds,
        normals, completionSamples, completionSampleNormals, goalPoses, goalCosts, targObjIdxs,
        obstacleCloudsAtStart, obstacleCloudsAtGoal)
      regraspPlanningTime.append(time() - startTime)
      print("Regrasp planner completed in {} seconds.".format(regraspPlanningTime[-1]))
      
      if len(pickPlaces) == 0:
        print("No regrasp plan found.")
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
      
      if showSteps:
        env.UnplotDescriptors()
        raw_input("End of episode.")
    
    nPlaced.append(int(success))
    totalTime = time() - totalStartTime
    data = {"nPlaced":nPlaced, "graspSuccess":graspSuccess, "graspAntipodal":graspAntipodal,
      "tempPlaceStable":tempPlaceStable, "planLength":planLength,
      "taskPlanningTime":taskPlanningTime, "regraspPlanningTime":regraspPlanningTime,
      "totalTime":totalTime}
    data.update(params)
    savemat(saveFileName, data)

if __name__ == "__main__":
  main()