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
from geom_pick_place.environment_blocks import EnvironmentBlocks

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
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/blocks_clouds_" + scenario
  placeDirectory = "/home/mgualti/Data/GeomPickPlace/blocks_places_" + scenario
  nExamples = 1000000 if scenario == "train" else 20000
  nObjects = 5
  
  # perception
  modelFileNameCompletion = "pcn_model_blocks.h5"
  modelFileNameSegmentation = "bonet_model_blocks.cptk"  
  nInputPointsCompletion = 1024
  nOutputPointsCompletion = 1024
  errorThresholdCompletion = 0.006
  nInputPointsSegmentation = 2048
  scoreThreshSegmentation = 0.80
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
  temporaryPlacePosition = array([0.25, 0.00, 0.00])
  nGraspSamples = 0
  nTempPlaceSamples = 10
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
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  halfAngleOfGraspFrictionCone = env.halfAngleOfGraspFrictionCone
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  Tsensor[0:2, 3] = point_cloud.WorkspaceCenter(sceneWorkspace)[0:2]
  sceneWorkspace[2][0] += env.GetTableHeight()
  sceneViewWorkspace[2][0] += env.GetTableHeight()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
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
        
    # sample places on all objects
    for i, cloud in enumerate(completedClouds):
      
      # exit early if enough examples
      if nPositive + nNegative >= nExamples:
        break
      
      # sample places
      supportingTriangles, triangleNormals, center = regraspPlanner.GetSupportSurfaces(cloud)      
      places, triangles = regraspPlanner.SamplePlaces(cloud, supportingTriangles, triangleNormals, center)
      
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
        Y = point_cloud.Transform(dot(place, placedObject.GetTransform()), placedObject.cloud)
        isStable = env.IsPlacementStable(Y, env.faceDistTol)
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