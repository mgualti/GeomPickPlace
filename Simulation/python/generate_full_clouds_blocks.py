#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
# scipy
from scipy.io import savemat
from numpy.random import seed
from scipy.spatial import cKDTree
from numpy import array, isnan, logical_not
# self
import point_cloud
from geom_pick_place.environment_blocks import EnvironmentBlocks

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test"
  randomSeed = 0 if scenario == "train" else 1
  
  # objects
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/blocks_clouds_" + scenario
  objectExtents = [(0.02, 0.07), (0.02, 0.07), (0.02, 0.07)]
  nObjects = 5000 if scenario == "train" else 1000
  
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
  
  # remove files in completion directory
  if os.path.exists(cloudDirectory):
    response = raw_input("Overwrite existing directory? (Y/N): ")
    if response.lower() != "y": return
    shutil.rmtree(cloudDirectory)
  os.mkdir(cloudDirectory)  
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveTable()
  env.RemoveRobot()
  env.RemoveFloatingHand()

  # RUN TEST =======================================================================================
  
  for i in xrange(nObjects):
      
    # generate a random rectangular block
    body = env.GenerateRandomBlock(objectExtents, "block-".format(i))
    
    # generate full cloud
    cloud, normals  = env.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
      add45DegViews = False, computeNormals = True, voxelSize = voxelSize)
    
    # occasionally, the normals calculation fails. replace with the nearest normal.
      if isnan(normals).any():
        nanIdx = sum(isnan(normals), axis = 1) > 0
        notNanIdx = logical_not(nanIdx)
        nanFreeTree = cKDTree(cloud[notNanIdx, :])
        nanFreeNormals = normals[notNanIdx, :]
        d, nearestIdx = nanFreeTree.query(cloud[nanIdx, :])
        normals[nanIdx, :] = nanFreeNormals[nearestIdx, :]
    
    # save data
    data = {"extents":body.extents, "cloud":cloud, "normals":normals}
    fileName = cloudDirectory + "/" + str(i) + ".mat"
    savemat(fileName, data)
    
    # optional visualization for debugging
    env.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud, normals, 3)
    printString = "Saved {}.".format(fileName)
    if showSteps: raw_input(printString)
    else: print(printString)
    
    # remove the object before loading the next one
    env.env.Remove(body)

if __name__ == "__main__":
  main()