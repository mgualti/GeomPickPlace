#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
from time import time
# scipy
from scipy.io import savemat
from numpy.random import choice, seed
from numpy import arange, array, concatenate, copy, dot, ones, zeros
import matplotlib
# self
import point_cloud
from bonet.bonet_model import BonetModel
from geom_pick_place.hand_descriptor import HandDescriptor
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.planner_bottles import PlannerBottles
from geom_pick_place.environment_bottles import EnvironmentBottles

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  
  scenario = "train"  
  if scenario == "train":
    randomSeed = 3
  elif scenario == "test":
    randomSeed = 4
  else:
    raise Exception("Unrecognized scenario {}.".format(scenario))
  
  # robot
  viewKeepout = 0.70
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [0.0, 1.00]]
  sceneViewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [0.005, 1.00]]
  
  # objects
  cloudDirectoryBottles = "/home/mgualti/Data/GeomPickPlace/bottles_clouds_" + scenario
  cloudDirectoryCoasters = "/home/mgualti/Data/GeomPickPlace/coasters_clouds_" + scenario
  graspDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_grasps_" + scenario
  nExamples = 1000000 if scenario == "train" else 20000
  nGraspSamples = 300
  nObjects = 2
  
  # perception
  modelFileNameCompletion = "pcn_model_bottles.h5"
  modelFileNameSegmentation = "bonet_model_bottles.cptk"  
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.007
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.75
  nInputPointsGraspPrediction = 128
  minPointsPerSegment = 1
  useGroundTruthSegmentation = False
  useGroundTruthCompletion = False
  useShapeCompletion = True
  
  # task planning
  nGoalOrientations = 1
  
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
  balanceData = True

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
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
    env.LoadInitialScene(nObjects, cloudDirectoryBottles, cloudDirectoryCoasters, sceneWorkspace)
    
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
    freeBottles, freeBottleProbs, placedBottles, freeCoasters, occupiedCoasters, _ = \
      taskPlanner.SegmentObjects(cloud, modelSegmentation, scoreThreshSegmentation,
      useGroundTruthSegmentation, nObjects)
    
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
    if useShapeCompletion:
      completedFreeBottles, completionProbs = taskPlanner.CompleteObjects(\
        freeBottles, modelCompletion, useGroundTruthCompletion)
    else:
      completedFreeBottles = []; completionProbs = []
      for i in xrange(len(freeBottles)):
        completedFreeBottles.append(copy(freeBottles[i]))
        completionProbs.append(ones(freeBottles[i].shape[0]))
        
    # compute normals for the target object
    normals = env.EstimateNormals(completedFreeBottles)
    
    # sample grasps on all objects
    pointCosts = regraspPlanner.GetPointCosts(
      freeBottles, completedFreeBottles, freeBottleProbs, completionProbs)
    grasps = regraspPlanner.PreSampleGraspsOnObjects(
      completedFreeBottles, normals, pointCosts, nGraspSamples)
    
    for i, cloud in enumerate(completedFreeBottles):
      for grasp in grasps[i]:
        
        # exit early if enough examples
        if nPositive + nNegative >= nExamples:
          break
        
        # identify object being grasped
        objsInHand = env.FindObjectsInHand(grasp)
        if len(objsInHand) != 1: continue
        objInHand = objsInHand[0]
        if objInHand not in env.bottleObjects: continue
        
        # check if grasp is antipodal, given ground truth shape
        X, N = point_cloud.Transform(dot(objInHand.GetTransform(), point_cloud.InverseTransform(\
          objInHand.cloudTmodel)), objInHand.cloud, objInHand.normals)
        isAntipodal, _ = env.IsAntipodalGrasp(grasp, X, N,
          env.cosHalfAngleOfGraspFrictionCone, env.graspContactWidth)
          
        # balance data
        if balanceData:
          if isAntipodal and nPositive >= nExamples / 2.0:
            continue
          elif not isAntipodal and nNegative >= nExamples / 2.0:
            continue
        
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