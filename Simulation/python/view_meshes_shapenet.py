#!/usr/bin/env python
'''Script for viewing mesh files with the OpenRAVE viewer.'''

# python
import os
import fnmatch
# scipy
from numpy import eye
# geom_pick_place
from geom_pick_place.environment_pick_place import EnvironmentPickPlace

def main():
  '''Entrypoint to the program.'''
  
  # PARAMETERS =====================================================================================  
  
  # objects
  meshDirectory = "/home/mgualti/Data/ShapeNetCoreV2/bottle"
  height = 0.15
  
  # environment
  floatingHandPose = eye(4)
  floatingHandPose[0:3, 0] = [ 0, 0, 1]
  floatingHandPose[0:3, 1] = [ 0, 1, 0]
  floatingHandPose[0:3, 2] = [-1, 0, 0]
  showViewer = True
  showWarnings = False

  # INITIALIZATION =================================================================================

  env = EnvironmentPickPlace(showViewer, showWarnings)
  env.RemoveRobot()
  env.RemoveSensor()
  env.RemoveTable()
  
  env.MoveFloatingHandToPose(floatingHandPose)
  
  # RUN TEST =======================================================================================
  
  fileNames = os.listdir(meshDirectory)
  fileNames = fnmatch.filter(fileNames, "*.obj")
  fileNames.sort(key = lambda fileName: int(fileName[:-4]))

  for fileName in fileNames:
    
    objectName = fileName[:-4]
    body, scale, T = env.LoadShapeNetObject(meshDirectory + "/" + fileName, height)
    raw_input(objectName)
    env.RemoveObjectSet([body])

if __name__ == "__main__":
  main()