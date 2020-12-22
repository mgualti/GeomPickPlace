#!/usr/bin/env python
'''Shows completed point clouds for a trained PCN model.'''

# python
import os
import fnmatch
# scipy
import matplotlib
from numpy.random import seed
from numpy import array
# self
import point_cloud
from geom_pick_place.pcn_model import PcnModel
from geom_pick_place.environment_packing import EnvironmentPacking

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test1"
  randomSeed = 0
  
  # sensor
  viewKeepout = 0.60
  viewWorkspace = [[-1.00, 1.00], [-1.00, 1.00], [-1.00, 1.00]]
  Tsensor = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, viewKeepout], [0, 0, 0, 1]])
  
  # objects
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/packing_clouds_" + scenario
  #sceneWorkspace = [[-0.18, 0.18], [-0.18, 0.18], [-1.00, 1.00]]
  sceneWorkspace = [[-0.001, 0.001], [-0.001, 0.001], [-1.00, 1.00]]
  nObjectPoses = 1
  
  # model
  modelFileName = "pcn_model_packing.h5"
  nInputPoints = 1024
  nOutputPoints = 1024
  errorThreshold = 0.006

  # visualization/saving
  showViewer = True
  showSteps = True
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveBox()
  env.RemoveTable()
  env.RemoveRobot()
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  Tsensor[2, 3] += env.GetTableHeight()
  sceneWorkspace[2][0] += env.GetTableHeight()
  viewWorkspace[2][0] += env.GetTableHeight()
  
  # initialize model
  model = PcnModel(nInputPoints, nOutputPoints, errorThreshold, -1, modelFileName)
  
  # set sensor pose
  env.MoveSensorToPose(Tsensor)

  # RUN TEST =======================================================================================
  
  cloudFileNames = os.listdir(cloudDirectory)
  cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
  
  for i, cloudFileName in enumerate(cloudFileNames):
    
    # load the next object
    obj = env.LoadObjectFromFullCloudFile(cloudDirectory, cloudFileName, "object-{}".format(i))
    
    # compute support faces
    triangles, normals, center = env.planner.GetSupportSurfaces(obj.cloud)
    
    for j in xrange(nObjectPoses):
      
      print("Object {}, pose {}.".format(i, j))
      
      # randomly perterb the object's orientation
      Tobject = env.PlaceObjectStablyOnTableAtRandom(obj, triangles, normals, center, sceneWorkspace)
      if showSteps: raw_input("Placed object.")
      
      # acquire top-view point cloud of the object
      Cpart = env.GetCloud(viewWorkspace)
      Ccomp = point_cloud.Transform(Tobject, obj.cloud)
      '''if showSteps:
        env.PlotCloud(Cpart)
        raw_input("Acquired partial cloud.")'''
        
      # predict completed cloud
      Cpred, correctProb = model.Predict(Cpart, env.GetTableHeight())
      print("Probability of completion correctness ranges from {} to {}.".format(
        min(correctProb), max(correctProb)))
      
      # visualize result
      if showSteps:
        env.PlotCloud(Cpart)
        raw_input("Partial cloud.")
        colors = matplotlib.cm.viridis(correctProb)
        env.PlotCloud(Cpred, colors)
        raw_input("Predicted complete cloud.")
        env.PlotCloud(Ccomp)
        raw_input("Complete cloud.")
    
    env.RemoveObjectSet([obj])

if __name__ == "__main__":
  main()