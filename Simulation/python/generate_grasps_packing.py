#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
from time import time
# scipy
from scipy.io import savemat
from numpy.random import choice, seed
from numpy import arange, array, concatenate, copy, ones, zeros
import matplotlib
# self
import point_cloud
from bonet.bonet_model import BonetModel
from geom_pick_place.hand_descriptor import HandDescriptor
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_packing import EnvironmentPacking

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  
  scenario = "train"  
  if scenario == "train":
    randomSeed = 3
  elif scenario == "test1":
    randomSeed = 4
  elif scenario == "test2":
    randomSeed = 5
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [0.0, 1.00]]
  sceneViewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  
  # objects
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/packing_clouds_" + scenario
  graspDirectory = "/home/mgualti/Data/GeomPickPlace/packing_grasps_" + scenario
  nExamples = 1000000 if scenario == "train" else 20000
  nGraspSamples = 200
  nObjects = 6
  
  # perception
  modelFileNameCompletion = "pcn_model_packing.h5"
  modelFileNameSegmentation = "bonet_model_packing.cptk"  
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.006
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.60
  nInputPointsGraspPrediction = 128
  minPointsPerSegment = 1
  useGroundTruthSegmentation = False
  useGroundTruthCompletion = False
  useShapeCompletion = True
  
  # cost factors
  taskCostFactor = 0
  regraspPlanStepCost = 0
  segmUncertCostFactor = 0
  compUncertCostFactor = 0
  graspUncertCostFactor = 0
  placeUncertCostFactor = 0
  antipodalCostFactor = 0
  insignificantCost = 0
  
  # regrasp planning
  temporaryPlacePosition = zeros(3)
  nTempPlaceSamples = 0
  nGoalPlaceSamples = 0
  maxSearchDepth = 0
  maxIterations = 0

  # visualization/saving
  showViewer = False
  showSteps = False
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveBox()
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  halfAngleOfGraspFrictionCone = env.halfAngleOfGraspFrictionCone
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  
  # initialize segmentation/completion models
  modelSegmentation = BonetModel(
    nInputPointsSegmentation, nObjects, deviceId, modelFileNameSegmentation)
  modelCompletion = PcnModel(nInputPointsCompletion, nOutputPointsCompletion,
    errorThresholdCompletion, deviceId, modelFileNameCompletion)
  
  # initialize planners
  regraspPlanner = PlannerRegrasp(env, temporaryPlacePosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor,
    regraspPlanStepCost, segmUncertCostFactor, compUncertCostFactor, graspUncertCostFactor,
    placeUncertCostFactor, antipodalCostFactor, insignificantCost, maxSearchDepth, maxIterations,
    minPointsPerSegment)
  
  # remove files in segmentation directory
  if os.path.exists(graspDirectory):
    response = raw_input("Overwrite {}? (Y/N): ".format(graspDirectory))
    if response.lower() != "y": return
    shutil.rmtree(graspDirectory)
  os.mkdir(graspDirectory)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  scene = 0; nPositive = 0; nNegative = 0
  startTime = time()
  
  while nPositive + nNegative < nExamples:
    
    scene += 1
    print("Scene {}. nPositive = {}. nNegative = {}.").format(scene, nPositive, nNegative)
  
    # load objects into scene
    env.LoadInitialScene(nObjects, cloudDirectory, sceneWorkspace)
    
    # clean visualization
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
      
    # segment objects
    segmentedClouds, segmentationProbs, _ = regraspPlanner.SegmentObjects(cloud, modelSegmentation,
      scoreThreshSegmentation, useGroundTruthSegmentation)
    
    if len(segmentedClouds) == 0:
      print("No segments found.")
      env.RemoveObjectNearestAnotherObject()
      continue
    
    if showSteps:
      env.PlotCloud(concatenate(segmentedClouds),
        matplotlib.cm.viridis(concatenate(segmentationProbs)))
      raw_input("Showing {} segments.".format(len(segmentedClouds)))
      
    # complete objects
    if useShapeCompletion:
      completedClouds, completionProbs = regraspPlanner.CompleteObjects(\
        segmentedClouds, modelCompletion, useGroundTruthCompletion)
    else:
      completedClouds = []; completionProbs = []
      for i in xrange(len(segmentedClouds)):
        completedClouds.append(copy(segmentedClouds[i]))
        completionProbs.append(ones(segmentedClouds[i].shape[0]))
    
    if showSteps:
      env.PlotCloud(concatenate(completedClouds),
        matplotlib.cm.viridis(concatenate(completionProbs)))
      raw_input("Showing {} completions".format(len(completedClouds)))
        
    # compute normals for the target object
    normals = env.EstimateNormals(completedClouds)
    
    # sample grasps on all objects
    pointCosts = regraspPlanner.GetPointCosts(
      segmentedClouds, completedClouds, segmentationProbs, completionProbs)
    grasps = regraspPlanner.PreSampleGraspsOnObjects(
      completedClouds, normals, pointCosts, nGraspSamples)
    
    for i, cloud in enumerate(completedClouds):
      for grasp in grasps[i]:
        
        # exit early if enough examples
        if nPositive + nNegative >= nExamples:
          break
        
        # identify object being grasped
        objsInHand = env.FindObjectsInHand(grasp)
        if len(objsInHand) != 1: continue
        objInHand = objsInHand[0]
        
        # check if grasp is antipodal, given ground truth shape
        X, N = point_cloud.Transform( \
          objInHand.GetTransform(), objInHand.cloud, objInHand.normals)
        isAntipodal, _ = env.IsAntipodalGrasp(grasp, X, N,
          env.cosHalfAngleOfGraspFrictionCone, env.graspContactWidth)
        
        # get observed points in hand frame
        desc = HandDescriptor(grasp)
        descPoints = desc.GetPointsInHandFrame(cloud)
        idx = choice(arange(descPoints.shape[0]), size = nInputPointsGraspPrediction)
        descPoints = descPoints[idx, :]
        
        # visualize result
        if showSteps:
          env.PlotDescriptors([desc])
          env.PlotCloud(point_cloud.Transform(grasp, descPoints))
          antipodalString = "is" if isAntipodal else "is not"
          raw_input("This grasp {} antipodal.".format(antipodalString))
        
        # save result
        descPoints = descPoints.astype('float32')
        data = {"cloud":descPoints, "correct":isAntipodal}
        fileName = graspDirectory + "/" + str(nPositive + nNegative) + ".mat"
        savemat(fileName, data)
        
        if isAntipodal: nPositive += 1
        else: nNegative += 1
  
  print("Found {} grasps, {} positive and {} negative (ratio {}%).").format(nPositive + nNegative,
    nPositive, nNegative, 100 * float(max(nPositive, nNegative)) / (nPositive + nNegative))
  print("Database size: {} MB.".format(((nPositive + nNegative) * 4 * (3 * \
    nInputPointsGraspPrediction + 1)) / 1024**2))
  print("Completed in {} hours.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()