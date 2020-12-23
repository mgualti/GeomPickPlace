#!/usr/bin/env python
'''Main script for executing the blocks experiment. Run this script once per arrangement episode.'''

# python
# scipy
import matplotlib
from numpy import array, concatenate, pi, zeros
# geom_pick_place
from place_blocks_params import Parameters
from bonet.bonet_model import BonetModel
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_blocks import PlannerBlocks
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_blocks import EnvironmentBlocks
# ros
import rospy
# robot
from robot.arm import Arm
from robot import utilities
from robot.gripper import Gripper
from robot.cloud_proxy import CloudProxy
from robot.motion_planner import MotionPlanner
# point_cloud
import point_cloud

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # task and environment
  nObjects = 6
  wholeWorkspace = [[0.10, 0.70], [-0.45, 0.45], [-0.20, 0.20]]
  clutterWorkspace = [[0.25, 0.56], [-0.14, 0.14], [0.01, 1.00]]
  rowWorkspace = [[0.20, 0.56], [-0.26, -0.16], [0.01, 1.00]]
  rowStart = point_cloud.WorkspaceCenter(rowWorkspace)
  rowStart[0] = rowWorkspace[0][1]
  
  # perception
  modelFileNameCompletion = "pcn_model_blocks.h5"
  modelFileNameSegmentation = "bonet_model_segmentation_packing.cptk"
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.005
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.50

  # general planning  
  preGraspOffset = params["preGraspOffset"]
  minPointsPerSegment = params["minPointsPerSegment"]
  
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
  planningTimeout = 20.0
  
  # visualization/saving
  isMoving = True
  showViewer = True
  showSteps = False
  showWarnings = False
  params = locals()
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveFloatingHand()
  env.RemoveSensor()
  env.SetHomeConfig(homeConfig)
  clutterWorkspace[2][0] += env.GetTableHeight()
  rowWorkspace[2][0] += env.GetTableHeight()
  regraspWorkspace[2][0] += env.GetTableHeight()
  rowStart[2] = env.GetTableHeight()
  regraspPosition[2] = env.GetTableHeight()  
  
  # initialize model
  modelSegmentation = BonetModel(nInputPointsSegmentation, nObjects, modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, modelFileNameCompletion)
  
  # initialize planners
  taskPlanner = PlannerBlocks(env, preGraspOffset, minPointsPerSegment, rowStart)
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
  
  # wait for ros to catch up
  rospy.sleep(1)
  
  # RUN TEST =======================================================================================
  
  # for tracking the placed objects, as they become an obstacle at the goal
  placedObjects = zeros((0, 3))
  isTemporaryPlace = False
  
  while not rospy.is_shutdown():
    
    env.UnplotDescriptors()
    env.UnplotCloud()
    
    if showSteps:
      raw_input("Ready.")
    
    # move arm to home
    gripper.Open()
    success = utilities.MoveToConfiguration(arm, motionPlanner, homeConfig)
    if not success:
      print("Failed to move the arm to home.")
      continue
    
    # acquire a point cloud for each part of the scene
    wholeCloud = utilities.GetCloud(cloudProxy, env, wholeWorkspace)
    cloud = point_cloud.FilterWorkspace(clutterWorkspace, wholeCloud)
    rowCloud = point_cloud.FilterWorkspace(rowWorkspace, wholeCloud)
    regraspCloud = point_cloud.FilterWorkspace(regraspWorkspace, wholeCloud)
    
    if placedObjects.shape[0] == 0:
      # (if nothing has been placed, then the row must be empty)
      rowCloud = zeros((0, 3))
    else:
      rowCloud = regraspPlanner.StackClouds([rowCloud, placedObjects])
      if rowCloud.shape[0] > 0:
        rowCloud = point_cloud.RemoveStatisticalOutliers(rowCloud, 30, 2)
      
    if regraspCloud.shape[0] > 0:
      regraspCloud = point_cloud.RemoveStatisticalOutliers(regraspCloud, 30, 2)
    
    if showSteps:
      if cloud.shape[0] > 0:
        env.PlotCloud(cloud)
        cloudProxy.PlotCloud(cloud)
        raw_input("Showing cloud of clutter.")
      if rowCloud.shape[0] > 0:
        env.PlotCloud(rowCloud)
        cloudProxy.PlotCloud(rowCloud)
        raw_input("Showing cloud of arranged blocks.")
      if regraspCloud.shape[0] > 0:
        env.PlotCloud(regraspCloud)
        cloudProxy.PlotCloud(regraspCloud)
        raw_input("Showing cloud of regrasp area.")
        
    if isTemporaryPlace:
      cloud = regraspCloud
    
    if cloud.shape[0] == 0:
      raw_input("The point cloud of the target object(s) is empty.")
      isTemporaryPlace = False
      continue
    
    # segment objects
    segmentedClouds, segmentationProbs = taskPlanner.SegmentObjects(cloud, modelSegmentation,
      clutterWorkspace, scoreThreshSegmentation, False)
      
    if len(segmentedClouds) == 0:
      raw_input("No semgents (i.e., objects) found.")
      continue
    
    if isTemporaryPlace:
      # (reduce to the largest segment found: there must be only 1 object here)
      maxPoints = -float('inf'); newSegmentedClouds = []; newSegmentationProbs = []
      for i, segmentedCloud in enumerate(segmentedClouds):
        if segmentedCloud.shape[0] > maxPoints:
          maxPoints = segmentedCloud.shape[0]
          newSegmentedClouds = [segmentedCloud]
          newSegmentationProbs = [segmentationProbs[i]]
      segmentedClouds = newSegmentedClouds
      segmentationProbs = newSegmentationProbs
  
    env.PlotCloud(concatenate(segmentedClouds),
      matplotlib.cm.viridis(concatenate(segmentationProbs)))
    if showSteps: raw_input("Showing {} segments.".format(len(segmentedClouds)))
    
    # complete objects
    completedClouds, completionProbs = taskPlanner.CompleteObjects(\
      segmentedClouds, modelCompletion, False)
      
    # compute normals for the target object
    normals = env.EstimateNormals(completedClouds)
    
    for i, cloud in enumerate(completedClouds):
      print("Completed cloud {} is higher than segmented cloud by {} mm.".format(
        i, 1000 * (max(cloud[:, 2]) - max(segmentedClouds[i][:, 2]))))
    env.PlotCloud(concatenate(completedClouds), matplotlib.cm.viridis(concatenate(completionProbs)))
    if showSteps: raw_input("Showing {} completions".format(len(completedClouds)))   
        
    # get goal poses from the task planner
    goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPoses(\
      completedClouds, normals, rowCloud)
    
    if goalPoses is None:
      env.RemoveObjectNearestAnotherObject()
      continue
    
    # get a sequence of grasp and place poses from the regrasp planner
    pickPlaces, goalIdx, targObjIdx = regraspPlanner.Plan(segmentedClouds, completedClouds, normals,
      goalPoses, targObjIdxs, segmentationProbs, completionProbs, goalCosts, rowCloud)      
    
    if pickPlaces is None:
      env.RemoveObjectNearestCloud(completedClouds[targObjIdx])
      continue
    
    env.MoveRobotToHome()
    env.PlotDescriptors(pickPlaces)
    if showSteps: raw_input("Showing regrasp plan.")
      
    # attempt to execute the first two steps, then re-plan
    homeToPreGraspTraj, graspConfig, homeToPrePlaceTraj, placeConfig = utilities.PlanPickPlace( \
      env, regraspPlanner, motionPlanner, pickPlaces[0], pickPlaces[1], rowCloud, completedClouds,
      targObjIdx, showSteps)
    
    if homeToPreGraspTraj is None:
      print("Failed to plan next pick-place.")
      continue
    
    success = utilities.ExecutePickPlace(arm, gripper, homeToPreGraspTraj, graspConfig,
      homeToPrePlaceTraj, placeConfig)
    
    # update the completed cloud of placed objects
    if success:
      isTemporaryPlace = len(pickPlaces) > 2
      if not isTemporaryPlace:
        placedObjects = regraspPlanner.StackClouds([placedObjects, point_cloud.Transform(
          goalPoses[goalIdx], completedClouds[targObjIdx])])      
    else:
      isTemporaryPlace = False

if __name__ == "__main__":
  main()