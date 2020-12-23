#!/usr/bin/env python
'''Main script for executing the packing experiment. Run this script once per arrangement episode.'''

# python
# scipy
import matplotlib
from numpy import array, concatenate, ones, pi, zeros
# geom_pick_place
from place_packing_params import Parameters
from bonet.bonet_model import BonetModel
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_packing import PlannerPacking
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_packing import EnvironmentPacking
# ros
import rospy
# robot
from robot.arm import Arm
from robot.gripper import Gripper
from robot.cloud_proxy import CloudProxy
from robot.motion_planner import MotionPlanner
from robot.utilities_packing import UtilitiesPacking
# point_cloud
import point_cloud

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # task and environment
  nObjects = 6
  wholeWorkspace = [[0.10, 0.90], [-0.45, 0.45], [-0.20, 0.20]]
  clutterWorkspace = [[0.30, 0.60], [-0.40, 0.20], [0.01, 1.00]]
  
  # perception
  modelFileNameCompletion = "pcn_model_packing.h5"
  modelFileNameSegmentation = "bonet_model_packing.cptk"
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.006
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.60 # 0.60 in simulation
  minPointsPerSegment = params["minPointsPerSegment"]

  # general planning 
  preGraspOffset = params["preGraspOffset"] 
  
  # task planning
  nInitialGraspSamples = params["nInitialGraspSamples"]
  loweringDelta = params["loweringDelta"]
  collisionCubeSize = params["collisionCubeSize"]
  nWorkers = params["nWorkers"]
  maxGoalsPerWorker = params["maxGoalsPerWorker"]
  maxTaskPlanningTime = params["maxTaskPlanningTime"]
  
  # regrasp planning
  regraspWorkspace = params["regraspWorkspace"]
  regraspPosition = params["regraspPosition"]
  nGraspSamples = params["nGraspSamples"]
  halfAngleOfGraspFrictionCone = params["halfAngleOfGraspFrictionCone"]
  nTempPlaceSamples = params["nTempPlaceSamples"]
  nGoalPlaceSamples = params["nGoalPlaceSamples"]
  maxSearchDepth = params["maxSearchDepth"]
  maxIterations = params["maxIterations"]
  
  # (cost factors)
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  segmUncertCostFactor = params["segmUncertCostFactor"]
  compUncertCostFactor = params["compUncertCostFactor"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  antipodalCostFactor = params["antipodalCostFactor"]
  insignificantCost = params["insignificantCost"]
  
  # motion planning
  homeConfig = array([-0.078, -1.989, 2.243, -1.827, -1.57, -0.023])  
  maxCSpaceJump = 8 * (pi / 180)
  nMotionPlanningAttempts = 5
  planningTimeout = 40.0
  
  # visualization/saving
  isMoving = True
  showViewer = True
  showSteps = False
  showWarnings = False
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveFloatingHand()
  env.RemoveSensor()
  env.SetHomeConfig(homeConfig)
  clutterWorkspace[2][0] += env.GetTableHeight()
  regraspWorkspace[2][0] += env.GetTableHeight()
  regraspPosition[2] = env.GetTableHeight()
  
  boxWorkspace = [(env.boxPosition[0] - env.boxExtents[0] / 2.0 + 0.02,
                   env.boxPosition[0] + env.boxExtents[0] / 2.0 - 0.02),
                  (env.boxPosition[1] - env.boxExtents[1] / 2.0 + 0.02,
                   env.boxPosition[1] + env.boxExtents[1] / 2.0 - 0.02),
                  (env.boxPosition[2] - env.boxExtents[2] / 2.0 + 0.02, 
                   env.boxPosition[2] + env.boxExtents[2] / 2.0 + 0.05)]
  
  # initialize model
  modelSegmentation = BonetModel(nInputPointsSegmentation, nObjects, -1, modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, -1, modelFileNameCompletion)
  
  # initialize planners
  taskPlanner = PlannerPacking(env, nInitialGraspSamples, loweringDelta, collisionCubeSize,
    nWorkers, maxGoalsPerWorker, maxTaskPlanningTime, minPointsPerSegment, preGraspOffset)
  regraspPlanner = PlannerRegrasp(env, regraspPosition, nGraspSamples, halfAngleOfGraspFrictionCone,
    nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor, regraspPlanStepCost, segmUncertCostFactor,
    compUncertCostFactor, graspUncertCostFactor, placeUncertCostFactor, antipodalCostFactor,
    insignificantCost, maxSearchDepth, maxIterations, minPointsPerSegment, preGraspOffset)
  
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
  utilities = UtilitiesPacking(arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner)
  
  # wait for ros to catch up
  rospy.sleep(1)
  
  # RUN TEST =======================================================================================
  
  # for remembering if the target object is in the regrasp area
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
    clutterCloud = point_cloud.FilterWorkspace(clutterWorkspace, wholeCloud)
    clutterCloud = point_cloud.RemoveStatisticalOutliers(clutterCloud, 30, 2)
    regraspCloud = point_cloud.FilterWorkspace(regraspWorkspace, wholeCloud)
    regraspCloud = point_cloud.RemoveStatisticalOutliers(regraspCloud, 30, 2)
    boxCloud = point_cloud.FilterWorkspace(boxWorkspace, wholeCloud)
    boxCloud = point_cloud.RemoveStatisticalOutliers(boxCloud, 30, 2)
    cloud = regraspCloud if isTemporaryPlace else clutterCloud
    
    if showSteps:
      if clutterCloud.shape[0] > 0:
        env.PlotCloud(clutterCloud)
        cloudProxy.PlotCloud(clutterCloud)
        raw_input("Showing cloud of clutter.")
      if boxCloud.shape[0] > 0:
        env.PlotCloud(boxCloud)
        cloudProxy.PlotCloud(boxCloud)
        raw_input("Showing cloud of box.")
      if regraspCloud.shape[0] > 0 and isTemporaryPlace:
        env.PlotCloud(regraspCloud)
        cloudProxy.PlotCloud(regraspCloud)
        raw_input("Showing cloud of regrasp area.")
    
    if boxCloud.shape[0] > 0:
      packingHeight = max(boxCloud[:, 2]) - env.GetTableHeight() - env.boxExtents[3]
      print("Packing height is {} cm.".format(packingHeight * 100))
    
    if cloud.shape[0] == 0:
      raw_input("The point cloud of the target object(s) is empty.")
      continue
    
    # segment objects
    segmentedClouds, segmentationProbs, _ = taskPlanner.SegmentObjects(
      cloud, modelSegmentation, scoreThreshSegmentation, False)
      
    if len(segmentedClouds) == 0:
      raw_input("No objects found in clutter.")
      continue
        
    if isTemporaryPlace and len(segmentedClouds) > 1:
      largestSegment = zeros((0, 3))
      largestSegmentProbs = zeros(0)
      for i in xrange(len(segmentedClouds)):
        if segmentedClouds[i].shape[0] > largestSegment.shape[0]:
          largestSegment = segmentedClouds[i]
          largestSegmentProbs = segmentationProbs[i]
      segmentedClouds = [largestSegment]
      segmentationProbs = [largestSegmentProbs]
  
    env.PlotCloud(concatenate(segmentedClouds), matplotlib.cm.viridis(concatenate(segmentationProbs)))
    if showSteps: raw_input("Showing {} segments.".format(len(segmentedClouds)))
    
    # complete objects
    completedClouds, completionProbs = taskPlanner.CompleteObjects(
      segmentedClouds, modelCompletion, False)
    env.PlotCloud(concatenate(completedClouds), matplotlib.cm.viridis(concatenate(completionProbs)))
    if showSteps: raw_input("Showing {} completed objects.".format(len(completedClouds)))    
    
    # compute surface normals for completed objects
    normals = env.EstimateNormals(completedClouds)
    
    # get goal placements from the task planner
    goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPoses(segmentedClouds,
      segmentationProbs, completedClouds, completionProbs, normals, boxCloud, regraspPlanner)
    
    if len(goalPoses) == 0:
      raw_input("No goal placements found.")
      continue
    
    # determine obstacles for regrasp planning
    obstacleCloudsAtStart = []
    if isTemporaryPlace:
      for targObjIdx in targObjIdxs:
        obstacleCloudsAtStart.append(clutterCloud)
    else:
      for targObjIdx in targObjIdxs:
        obstacleCloudsAtStart.append(regraspPlanner.StackClouds(completedClouds, [targObjIdx]))
      
    obstacleCloudsAtGoal = []
    for targObjIdx in targObjIdxs:
      obstacleCloudsAtGoal.append(boxCloud)
    
    # get a sequence of grasp and place poses from the regrasp planner
    pickPlaces, goalIdx, targObjIdx = regraspPlanner.Plan(segmentedClouds, segmentationProbs,
      completedClouds, completionProbs, normals, goalPoses, goalCosts, targObjIdxs,
      obstacleCloudsAtStart, obstacleCloudsAtGoal)
    
    if len(pickPlaces) == 0:
      raw_input("No regrasp plan found.")
      continue
    
    env.MoveRobotToHome()
    env.PlotDescriptors(pickPlaces)
    if showSteps: raw_input("Showing regrasp plan.")
    
    # search for a motion plan, continuing regrasp planner if one is not found
    for motionPlanningAttempt in xrange(nMotionPlanningAttempts):
      
      print("Searching for a motion plan, attempt {}.".format(motionPlanningAttempt + 1))
      homeToPreGraspTraj, preGraspToHomeTraj, graspConfig, homeToPrePlaceTraj, prePlaceToHomeTraj, \
        placeConfig = utilities.PlanPickPlace(pickPlaces[0], pickPlaces[1],
        completedClouds[targObjIdx], obstacleCloudsAtStart[goalIdx], obstacleCloudsAtGoal[goalIdx],
        showSteps)
      
      if homeToPreGraspTraj is not None or motionPlanningAttempt == nMotionPlanningAttempts - 1:
        break
      
      print("Continuing regrasp planner...")
      pickPlaces, goalIdx, targObjIdx = regraspPlanner.Continue()
      
      if len(pickPlaces) == 0:
        print("No regrasp plan found.")
        break
    
      env.PlotDescriptors(pickPlaces)
      if showSteps: raw_input("Showing regrasp plan.")
      
    if homeToPreGraspTraj is None:
      print("Failed to plan next pick-place.")
      continue
    
    # execute the first two steps, then re-plan
    success = utilities.ExecutePickPlace(homeToPreGraspTraj, preGraspToHomeTraj, graspConfig,
      homeToPrePlaceTraj, prePlaceToHomeTraj, placeConfig)
    
    # update the completed cloud of placed objects
    if success:
      isTemporaryPlace = len(pickPlaces) > 2     
    else:
      isTemporaryPlace = False

if __name__ == "__main__":
  main()