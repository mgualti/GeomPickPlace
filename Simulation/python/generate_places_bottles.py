#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
from time import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from numpy import argmin, array, concatenate, copy, dot, ones, sum, power, reshape, tile
import matplotlib
# self
import point_cloud
from bonet.bonet_model import BonetModel
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
    randomSeed = 5
  elif scenario == "test":
    randomSeed = 6
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
  placeDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_places_" + scenario
  nExamples = 1000000 if scenario == "train" else 20000
  nObjects = 2
  
  # perception
  modelFileNameCompletion = "pcn_model_bottles.h5"
  modelFileNameSegmentation = "bonet_model_bottles.cptk"  
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.007
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.75
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
  temporaryPlacePosition = array([0.25, 0.00, 0.00])
  nGraspSamples = 0
  nTempPlaceSamples = 25
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
  
  faceDistTol = env.faceDistTol / 2.0
  halfAngleOfGraspFrictionCone = env.halfAngleOfGraspFrictionCone
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
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
  if os.path.exists(placeDirectory):
    response = raw_input("Overwrite {}? (Y/N): ".format(placeDirectory))
    if response.lower() != "y": return
    shutil.rmtree(placeDirectory)
  os.mkdir(placeDirectory)
  
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
        
    # sample places on all objects
    for i, cloud in enumerate(completedFreeBottles):
      
      # exit early if enough examples
      if nPositive + nNegative >= nExamples:
        break
      
      # sample places
      supportingTriangles, triangleNormals, center = regraspPlanner.GetSupportSurfaces(cloud)      
      places, triangles = regraspPlanner.SamplePlaces(
        cloud, supportingTriangles, triangleNormals, center)
      
      # find actual object
      actualCenters = env.GetObjectCentroids(env.unplacedObjects)
      centers = tile(center, (actualCenters.shape[0], 1))
      objIdx = argmin(power(sum(actualCenters - centers, axis = 1), 2))
      placedObject = env.unplacedObjects[objIdx]
      
      for place in places:
        
        # exit early if enough examples
        if nPositive + nNegative >= nExamples:
          break
        
        # check if place is stable, given ground truth shape
        Y = point_cloud.Transform(dot(place, dot(placedObject.GetTransform(),
          point_cloud.InverseTransform(placedObject.cloudTmodel))), placedObject.cloud)
        isStable = env.IsPlacementStable(Y, faceDistTol)
        descPoints = point_cloud.Transform(place, cloud) - tile(reshape(temporaryPlacePosition, \
          (1, 3)), (cloud.shape[0], 1))
          
        # balance data
        if balanceData:
          if isStable and nPositive >= nExamples / 2.0:
            continue
          elif not isStable and nNegative >= nExamples / 2.0:
            continue
        
        # visualize result
        if showSteps:
          visCloud = copy(descPoints)
          visCloud += tile(reshape(temporaryPlacePosition, (1, 3)), (visCloud.shape[0], 1))
          env.PlotCloud(visCloud)
          originalObjectPose = placedObject.GetTransform()
          placedObject.SetTransform(dot(place, placedObject.GetTransform()))
          isStableString = "is" if isStable else "is not"
          raw_input("This place {} stable.".format(isStableString))
          placedObject.SetTransform(originalObjectPose)
        
        # save result
        descPoints = descPoints.astype('float32')
        data = {"cloud":descPoints, "correct":isStable}
        fileName = placeDirectory + "/" + str(nPositive + nNegative) + ".mat"
        savemat(fileName, data)
        
        if isStable: nPositive += 1
        else: nNegative += 1
  
  print("Found {} places, {} positive and {} negative (ratio {}%).").format(nPositive + nNegative,
    nPositive, nNegative, 100 * float(max(nPositive, nNegative)) / (nPositive + nNegative))
  print("Database size: {} MB.".format(((nPositive + nNegative) * 4 * (3 * \
    nOutputPointsCompletion + 1)) / 1024**2))
  print("Completed in {} hours.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()