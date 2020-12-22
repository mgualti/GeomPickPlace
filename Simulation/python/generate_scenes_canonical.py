#!/usr/bin/env python
'''Generates initial scenes for the canonical task and saves those scenes to files.'''

# python
import os
import shutil
from time import time
# scipy
from numpy.random import seed
# self
from geom_pick_place.environment_canonical import EnvironmentCanonical

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test2"
  randomSeed = 0
  
  # environment
  sceneWorkspace = [[0.30, 0.48], [-0.18, 0.18], [0.01, 1.00]]
  
  # objects
  nObjects = 5
  nEpisodes = 5000  
  cloudDirectory = "/home/mgualti/Data/GeomPickPlace/packing_clouds_" + scenario
  sceneDirectory = "/home/mgualti/Data/GeomPickPlace/canonical_scenes_" + scenario + "_" + \
    str(nObjects) + "objects"
  
  # visualization/saving
  showViewer = False
  showSteps = False
  showWarnings = False

  # INITIALIZATION =================================================================================
  
  # set random seed
  seed(randomSeed)
  
  # initialize environment
  env = EnvironmentCanonical(showViewer, showWarnings)
  env.RemoveSensor()
  env.RemoveFloatingHand()
  
  # account for workspace geometry
  sceneWorkspace[2][0] += env.GetTableHeight()
  
  # remove files in segmentation directory
  if os.path.exists(sceneDirectory):
    response = raw_input("Overwrite {}? (Y/N): ".format(sceneDirectory))
    if response.lower() != "y": return
    shutil.rmtree(sceneDirectory)
  os.mkdir(sceneDirectory)
  
  # RUN TEST =======================================================================================
  
  startTime = time()
  
  for episode in xrange(nEpisodes):
  
    env.GenerateInitialScene(nObjects, cloudDirectory, sceneWorkspace)
    env.SaveScene(sceneDirectory + "/{}.mat".format(episode))
    
    progressString = "Scene {}.".format(episode + 1)
    if showSteps: raw_input(progressString)
    else: print(progressString)
    
  print("Completed in {} hours.".format((time() - startTime) / 3600.0))

if __name__ == "__main__":
  main()