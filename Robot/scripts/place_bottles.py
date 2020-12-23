#!/usr/bin/env python
'''Main script for executing the bottles experiment. Run this script once per arrangement episode.'''

# python
# scipy
import matplotlib
from numpy import array, concatenate, ones, pi, zeros
# geom_pick_place
from place_bottles_params import Parameters
from bonet.bonet_model import BonetModel
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_bottles import PlannerBottles
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_bottles import EnvironmentBottles
# ros
import rospy
# robot
from robot.arm import Arm
from robot.gripper import Gripper
from robot.cloud_proxy import CloudProxy
from robot.motion_planner import MotionPlanner
from robot.utilities_bottles import UtilitiesBottles
# point_cloud
import point_cloud

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # task and environment
  nObjects = 2
  wholeWorkspace = [[0.10, 0.70], [-0.45, 0.45], [-0.20, 0.20]]
  clutterWorkspace = [[0.30, 0.66], [-0.18, 0.18], [0.01, 1.00]]
  
  # perception
  modelFileNameCompletion = "pcn_model_bottles.h5"
  modelFileNameSegmentation = "bonet_model_bottles.cptk"
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.007
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.25

  # general planning 
  preGraspOffset = params["preGraspOffset"]
  minPointsPerSegment = params["minPointsPerSegment"]
  
  # task planning
  nGoalOrientations = 30
  
  # regrasp planning
  regraspWorkspace = params["regraspWorkspace"]
  regraspPosition = params["regraspPosition"]
  nGraspSamples = params["nGraspSamples"]
  graspFrictionCone = params["graspFrictionCone"]
  graspContactWidth = params["graspContactWidth"]
  nTempPlaceSamples = params["nTempPlaceSamples"]
  nGoalPlaceSamples = params["nGoalPlaceSamples"]
  maxSearchDepth = params["maxSearchDepth"]
  maxTimeToImprovePlan = params["maxTimeToImprovePlan"]
  maxTimeToPlan = params["maxTimeToPlan"]
  
  # (cost factors)
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  segmUncertCostFactor = params["segmUncertCostFactor"]
  compUncertCostFactor = params["compUncertCostFactor"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  insignificantCost = params["insignificantCost"]
  
  # motion planning
  homeConfig = array([-0.078, -1.989, 2.243, -1.827, -1.57, -0.023])
  maxCSpaceJump = 8 * (pi / 180)
  planningTimeout = 40.0
  
  # visualization/saving
  isMoving = True
  showViewer = True
  showSteps = True
  showWarnings = False
  params = locals()
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
  env.RemoveFloatingHand()
  env.RemoveSensor()
  env.SetHomeConfig(homeConfig)
  clutterWorkspace[2][0] += env.GetTableHeight()
  regraspWorkspace[2][0] += env.GetTableHeight()
  regraspPosition[2] = env.GetTableHeight()  
  
  # initialize model
  modelSegmentation = BonetModel(nInputPointsSegmentation, 2 * nObjects, -1, modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, -1, modelFileNameCompletion)
  
  # initialize planners
  taskPlanner = PlannerBottles(env, nGoalOrientations, preGraspOffset, minPointsPerSegment)
  regraspPlanner = PlannerRegrasp(env, preGraspOffset, minPointsPerSegment, regraspPosition,
    nGraspSamples, graspFrictionCone, graspContactWidth, nTempPlaceSamples, nGoalPlaceSamples,
    taskCostFactor, regraspPlanStepCost, segmUncertCostFactor, compUncertCostFactor,
    graspUncertCostFactor, placeUncertCostFactor, insignificantCost, maxSearchDepth,
    maxTimeToImprovePlan, maxTimeToPlan)
  
  # initialize ros
  rospy.init_node("GeomPickPlaceRobot")
  
  # initialize arm, gripper, and motion planning
  arm = Arm(env, isMoving)
  gripper = Gripper(isMoving)
  motionPlanner = MotionPlanner(maxCSpaceJump, planningTimeout)
  
  # initialize depth sensor
  cloudProxy = CloudProxy()
  cloudProxy.InitRos()
  
  # initialize utilities
  utilities = UtilitiesBottles(arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner)
  
  # wait for ros to catch up
  rospy.sleep(1)
  
  # RUN TEST =======================================================================================
  
  # for tracking the placed objects, as they become an obstacle at the goal
  isTemporaryPlace = False
  
  while not rospy.is_shutdown():
    
    env.UnplotDescriptors()
    env.UnplotCloud()
    
    if showSteps:
      raw_input("Ready.")
    
    # move arm to home
    gripper.Open()
    success = utilities.MoveToConfiguration(homeConfig)
    if not success:
      print("Failed to move the arm to home.")
      continue
    
    # acquire a point cloud for each part of the scene
    wholeCloud = utilities.GetCloud(wholeWorkspace)
    cloud = point_cloud.FilterWorkspace(clutterWorkspace, wholeCloud)
    regraspCloud = point_cloud.FilterWorkspace(regraspWorkspace, wholeCloud) if isTemporaryPlace \
      else zeros((0, 3))
    
    if showSteps:
      if cloud.shape[0] > 0:
        env.PlotCloud(cloud)
        cloudProxy.PlotCloud(cloud)
        raw_input("Showing cloud of clutter.")
      if regraspCloud.shape[0] > 0:
        env.PlotCloud(regraspCloud)
        cloudProxy.PlotCloud(regraspCloud)
        raw_input("Showing cloud of regrasp area.")
    
    if cloud.shape[0] == 0:
      raw_input("The point cloud of the target object(s) is empty.")
      isTemporaryPlace = False
      continue
    
    # segment objects
    freeBottles, freeBottleProbs, placedBottles, freeCoasters, occupiedCoasters = \
      taskPlanner.SegmentObjects(cloud, modelSegmentation, clutterWorkspace,
      scoreThreshSegmentation, False, nObjects)
        
    if isTemporaryPlace:
      freeBottles = [regraspCloud]
      freeBottleProbs = ones(regraspCloud.shape[0])
      placedBottles = placedBottles + freeBottles
      
    if len(freeBottles) == 0:
      raw_input("No free bottles found.")
      continue
    
    if len(freeCoasters) == 0:
      raw_input("No free coasters found.")
      continue
  
    env.PlotCloud(concatenate(freeBottles), matplotlib.cm.viridis(concatenate(freeBottleProbs)))
    if showSteps:
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
    
    # complete obstacles
    if len(placedBottles) > 0:
      placedBottles, placedBottleProbs = taskPlanner.CompleteObjects( \
        placedBottles, modelCompletion, False)
      env.PlotCloud(concatenate(placedBottles),
        matplotlib.cm.viridis(concatenate(placedBottleProbs)))
      if showSteps: raw_input("Showing {} completed obstacles".format(len(placedBottles)))    
    
    # complete objects
    completedFreeBottles, completionProbs = taskPlanner.CompleteObjects(\
      freeBottles, modelCompletion, False)
    
    env.PlotCloud(concatenate(completedFreeBottles),
      matplotlib.cm.viridis(concatenate(completionProbs)))
    if showSteps: raw_input("Showing {} completions".format(len(completedFreeBottles)))
    
    # compute normals for the target object
    normals = env.EstimateNormals(completedFreeBottles)
    
    # get goal poses from the task planner
    goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPoses(completedFreeBottles, freeCoasters)
    
    if goalPoses is None:
      raw_input("No goal placements found.")
      continue
    
    # determine obstacles for regrasp planning
    obstacleClouds = []
    for targObjIdx in targObjIdxs:
      obstacleClouds.append(regraspPlanner.StackClouds(\
        completedFreeBottles + placedBottles + freeCoasters + occupiedCoasters, [targObjIdx]))
    
    # get a sequence of grasp and place poses from the regrasp planner
    pickPlaces, goalIdx, targObjIdx = regraspPlanner.Plan(freeBottles, freeBottleProbs,
      completedFreeBottles, completionProbs, normals, goalPoses, goalCosts, targObjIdxs,
      obstacleClouds, obstacleClouds)
    
    if pickPlaces is None:
      raw_input("No regrasp plan found.")
      continue
    
    env.MoveRobotToHome()
    env.PlotDescriptors(pickPlaces)
    if showSteps: raw_input("Showing regrasp plan.")
      
    # attempt to execute the first two steps, then re-plan
    obstacleCloud = regraspPlanner.StackClouds(
      completedFreeBottles + placedBottles + freeCoasters + occupiedCoasters, [targObjIdx])
    homeToPreGraspTraj, preGraspToHomeTraj, graspConfig, homeToPrePlaceTraj, prePlaceToHomeTraj, \
      placeConfig = utilities.PlanPickPlace(pickPlaces[0], pickPlaces[1], goalPoses[goalIdx],
      completedFreeBottles[targObjIdx], obstacleCloud, showSteps)
    
    if homeToPreGraspTraj is None:
      print("Failed to plan next pick-place.")
      continue
    
    success = utilities.ExecutePickPlace(homeToPreGraspTraj, preGraspToHomeTraj, graspConfig,
      homeToPrePlaceTraj, prePlaceToHomeTraj, placeConfig)
    
    # update the completed cloud of placed objects
    if success:
      isTemporaryPlace = len(pickPlaces) > 2     
    else:
      isTemporaryPlace = False

if __name__ == "__main__":
  main()