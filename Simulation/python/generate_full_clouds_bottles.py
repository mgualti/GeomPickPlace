#!/usr/bin/env python
'''Generates full point clouds for bottle meshes (at different, uniformly sampled scales) and for
   coasters (sampled uniformly at random).'''

# python
import os
import shutil
import fnmatch
from time import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from scipy.spatial import cKDTree
from numpy import array, eye, isnan, linspace, logical_not, sum
# self
import point_cloud
from geom_pick_place.environment_bottles import EnvironmentBottles

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "train"
  randomSeed = 0 if scenario == "train" else 1
  
  # objects
  bottleMeshDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_models_" + scenario
  bottleCloudDirectory = "/home/mgualti/Data/GeomPickPlace/bottles_clouds_" + scenario
  coasterCloudDirectory = "/home/mgualti/Data/GeomPickPlace/coasters_clouds_" + scenario
  bottleHeightRange = [0.09, 0.22]
  nScalesPerBottle = 14
  nCoasters = 5000 if scenario == "train" else 1000
  add45DegViews = False
  
  # view
  viewCenter = array([0, 0, 0])
  viewKeepout = 0.60
  viewWorkspace = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
  voxelSize = 0.002

  # visualization/saving
  showViewer = False
  showSteps = False
  plotImages = False
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
  env.RemoveTable()
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  # other  
  bottleHeights = linspace(bottleHeightRange[0], bottleHeightRange[1], nScalesPerBottle)
  
  # remove files in existing directories
  if os.path.exists(bottleCloudDirectory) or os.path.exists(coasterCloudDirectory):
    response = raw_input("Overwrite existing directories? (Y/N): ")
    if response.lower() != "y": return
    if os.path.exists(bottleCloudDirectory): shutil.rmtree(bottleCloudDirectory)
    if os.path.exists(coasterCloudDirectory): shutil.rmtree(coasterCloudDirectory)
  os.mkdir(bottleCloudDirectory); os.mkdir(coasterCloudDirectory)
  
  # COASTERS =======================================================================================
  
  startTime = time()  
  
  for i in xrange(nCoasters):
    
    # load mesh into environment
    body = env.LoadRandomSupportObject("support-{}".format(i))
    
    # generate full cloud
    cloud = env.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
      add45DegViews = add45DegViews, computeNormals = False, voxelSize=voxelSize)
    
    # save data
    scale = 1.0
    cloudTmodel = eye(4)
    cloud = cloud.astype("float32")
    data = {"extents":body.extents, "scale":scale, "cloudTmodel":cloudTmodel, "cloud":cloud}
    fileName = coasterCloudDirectory + "/" + str(i) + ".mat"
    savemat(fileName, data)
    
    # optional visualization for debugging
    env.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud)
    printString = "Saved {}.".format(fileName)
    if showSteps: raw_input(printString)
    else: print(printString)
    
    # remove the object before loading the next one
    env.ResetScene()
    
  coasterGenerationTime = time() - startTime

  # BOTTLES ========================================================================================
  
  # locate mesh files
  meshFileNames = os.listdir(bottleMeshDirectory)
  meshFileNames = fnmatch.filter(meshFileNames, "*.obj")
  meshFileNames.sort(key = lambda fileName: int(fileName[:-4]))
  
  nMeshesUsed = 0; nHeightsUsed = 0; startTime = time()
  
  for i, meshFileName in enumerate(meshFileNames):
    
    usedMesh = False
    for j, height in enumerate(bottleHeights):
      
      # load mesh into environment
      body, scale, T = env.LoadShapeNetObject(bottleMeshDirectory + "/" + meshFileName, height)
      
      # generate full cloud
      cloud, normals  = env.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
        add45DegViews = add45DegViews, computeNormals = True, voxelSize = voxelSize)
        
      # occasionally, the normals calculation fails. replace with the nearest normal.
      if isnan(normals).any():
        nanIdx = sum(isnan(normals), axis = 1) > 0
        notNanIdx = logical_not(nanIdx)
        nanFreeTree = cKDTree(cloud[notNanIdx, :])
        nanFreeNormals = normals[notNanIdx, :]
        d, nearestIdx = nanFreeTree.query(cloud[nanIdx, :])
        normals[nanIdx, :] = nanFreeNormals[nearestIdx, :]
        
      # test if the entire object is too small or large for the gripper
      if not env.IsObjectWidthInLimits(cloud):
        if showSteps: raw_input("Object does not fit in gripper.")
        env.ResetScene()
        continue
      
      # save data
      cloud = cloud.astype("float32")
      normals = normals.astype("float32")
      data = {"meshFileName":bottleMeshDirectory + "/" + meshFileName, "height":height,
        "scale":scale, "cloudTmodel":T, "cloud":cloud, "normals":normals}
      fileName = bottleCloudDirectory + "/" + str(bottleHeights.size * i + j) + ".mat"
      savemat(fileName, data)
      usedMesh = True
      nHeightsUsed += 1
      
      # optional visualization for debugging
      env.PlotCloud(cloud)
      if plotImages: point_cloud.Plot(cloud, normals, 10)
      printString = "Saved {}.".format(fileName)
      if showSteps: raw_input(printString)
      else: print(printString)
      
      # remove the object before loading the next one
      env.ResetScene()
      
    nMeshesUsed += usedMesh
  
  bottleGenerationTime = time() - startTime
  
  # PRINT RESULT ===================================================================================
  
  print("Took {} hours to generate full clouds of coasters.".format(coasterGenerationTime / 3600.0))
  print("Took {} hours to generate full clouds of bottles.".format(bottleGenerationTime / 3600.0))  
  print("Used {}/{} bottle meshes; saved {} bottle clouds.".format(
    nMeshesUsed, len(meshFileNames), nHeightsUsed))

if __name__ == "__main__":
  main()