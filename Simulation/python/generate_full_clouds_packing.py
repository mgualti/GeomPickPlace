#!/usr/bin/env python
'''Generates full point clouds for randomly scaled meshes.'''

# python
import os
import shutil
import fnmatch
from time import time
# scipy
from scipy.io import savemat
from scipy.spatial import cKDTree
from numpy.random import choice, randint, seed, uniform
from numpy import all, arange, array, cos, eye, isnan, logical_not, pi, repeat, sum, tile, zeros
# self
import point_cloud
from geom_pick_place.hand_descriptor import HandDescriptor
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_packing import EnvironmentPacking

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "train"
  
  if scenario == "train":
    randomSeed = 0
  elif scenario == "test1":
    randomSeed = 1
  elif scenario == "test2":
    randomSeed = 2
  
  # objects
  meshDirectory = "/home/mgualti/Data/GeomPickPlace/packing_models_" + scenario
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/packing_clouds_" + scenario
  
  scaleRange = {"airplane":(0.06, 0.15), "boat":(0.06, 0.15), "bottle":(0.09, 0.22),
    "bowl":(0.06, 0.15), "box":(0.05, 0.16), "car":(0.06, 0.14), "dinosaur":(0.06, 0.18),
    "mug":(0.05, 0.12), "stapler":(0.07, 0.16), "wine_glass":(0.08, 0.18)}
  
  nClouds = 7000 if scenario == "train" else 1000
  
  # view
  voxelSize = 0.002
  viewKeepout = 0.60
  add45DegViews = True
  viewCenter = array([0, 0, 0])
  viewWorkspace = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
  
  # filtering
  minPoints = 300  
  nGraspSamples = 500
  graspContactWidth = 0.005
  graspFrictionCone = 10.0 * pi / 180
  minNegApproachDotGavity = cos(102.0 * pi / 180)

  # visualization/saving
  showViewer = False
  showSteps = False
  plotImages = False
  showWarnings = False

  # INITIALIZATION =================================================================================

  # set random seed
  seed(randomSeed)

  # initialize environment
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveBox()
  env.RemoveRobot()
  env.RemoveTable()
  env.RemoveFloatingHand()
  
  # initialize regrasp planner
  plannerRegrasp = PlannerRegrasp(env, zeros(3), nGraspSamples, graspFrictionCone,
    graspContactWidth, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  
  # remove files in clouds directory
  if os.path.exists(cloudDirectory):
    response = raw_input("Overwrite {}? (Y/N): ".format(cloudDirectory))
    if response.lower() != "y": return
    shutil.rmtree(cloudDirectory)
  os.mkdir(cloudDirectory)
  
  # RUN TEST =======================================================================================
  
  # find mesh file names
  categoryNames = os.listdir(meshDirectory)
  meshFileNames = {}
  
  for categoryName in categoryNames:
    names = os.listdir(meshDirectory + "/" + categoryName)
    meshFileNames[categoryName] = fnmatch.filter(names, "*.ply")
  
  i = 0
  startTime = time()
  
  while i < nClouds:
    
    # select category, mesh, and scale uniformly at random
    categoryName = categoryNames[i % len(categoryNames)]
    meshFileName = meshFileNames[categoryName][randint(len(meshFileNames[categoryName]))]
    scale = uniform(scaleRange[categoryName][0], scaleRange[categoryName][1])
    
    # load mesh into environment
    env.ResetScene()
    env.Load3DNetObject(meshDirectory + "/" + categoryName + "/" + meshFileName, scale)
    
    # generate full cloud
    cloud, normals  = env.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
      add45DegViews = add45DegViews, computeNormals = True, voxelSize = voxelSize)
      
    # test if enough points were found
    if cloud.shape[0] < minPoints:
      if showSteps: raw_input("Not enough points sampled on the object's surface.")
      else: print("Not enough points sampled on the object's surface.")
      continue
      
    # occasionally, the normals calculation fails. replace with the nearest normal.
    if isnan(normals).any():
      nanIdx = sum(isnan(normals), axis = 1) > 0
      notNanIdx = logical_not(nanIdx)
      if sum(notNanIdx) < minPoints:
        print("Not enough non-NaN normals.")
        continue
      nanFreeTree = cKDTree(cloud[notNanIdx, :])
      nanFreeNormals = normals[notNanIdx, :]
      d, nearestIdx = nanFreeTree.query(cloud[nanIdx, :])
      normals[nanIdx, :] = nanFreeNormals[nearestIdx, :]
      
    # optional visualization for debugging
    env.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud, normals, 10)
      
    # Test if the object can be grasped from any placement. We use the following heuristic: if for
    # every stable placement there is a grasp approaching from above the table, we assume the object
    # can be grasped from any placement. Otherwise, we assume there are some placements for which
    # the object cannot be grasped.
    
    # find grasps
    idx = choice(arange(cloud.shape[0]), 1024)
    downsampledCloud = cloud[idx]; downsampledNormals = normals[idx]
    pairs, binormals = plannerRegrasp.GetAntipodalPairsOfPoints(downsampledCloud, downsampledNormals)
    grasps, _ = plannerRegrasp.SampleGrasps(downsampledCloud, pairs, binormals)
    
    if showSteps:
      descriptors = []
      for grasp in grasps:
        descriptors.append(HandDescriptor(grasp))
      env.PlotDescriptors(descriptors)
      
    if len(grasps) == 0:
      if showSteps: raw_input("No grasps found.")
      else: print("No grasps found.")
      continue
      
    # find stable placements
    _, placementNormals, _ = plannerRegrasp.GetSupportSurfaces(downsampledCloud)
      
    # check that the dot product between every placement normal and some negative approach direction
    # is less than the given threshold
    negativeApproach = zeros((len(grasps), 3))
    for j, grasp in enumerate(grasps):
      negativeApproach[j, :] = grasp[0:3, 2]
    
    A = tile(negativeApproach, (placementNormals.shape[0], 1))
    N = repeat(placementNormals, negativeApproach.shape[0], axis = 0)
    D = sum(N * A, axis = 1)
    
    isGraspable = True
    for j in xrange(placementNormals.shape[0]):
      if all(D[negativeApproach.shape[0] * j : negativeApproach.shape[0] * (j + 1)] > \
        minNegApproachDotGavity):
        isGraspable = False
        break
    
    if not isGraspable:
      if showSteps: raw_input("Object not graspable.")
      else: print ("Object not graspable.")
      continue
      
    # save data
    cloud = cloud.astype("float32")
    normals = normals.astype("float32")
    cloudTmodel = eye(4, dtype = "float32")
    data = {"meshFileName":meshDirectory + "/" + categoryName + "/" + meshFileName, "scale":scale,
      "cloudTmodel":cloudTmodel, "cloud":cloud, "normals":normals}
    fileName = cloudDirectory + "/" + str(i) + ".mat"
    savemat(fileName, data)
    
    printString = "Saved {} with {} points".format(fileName, cloud.shape[0])
    if showSteps: raw_input(printString)
    else: print(printString)
    
    # indicate the cloud was saved successfully
    i += 1
  
  print("Took {} hours to generate full clouds.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()