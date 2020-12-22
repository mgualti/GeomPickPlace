#!/usr/bin/env python
'''TODO'''

# python
import sys
from multiprocessing.connection import Listener
# scipy
# tensorflow
import tensorflow
# self
from place_blocks_sp_params import Parameters
from geom_pick_place.planner_regrasp_sp import PlannerRegraspSP
from geom_pick_place.environment_blocks import EnvironmentBlocks

# uncomment when profiling
#import os; os.chdir("/home/mgualti/GeomPickPlace/Simulation")

def main(wid):
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # perception
  modelFileNameGraspPrediction = params["modelFileNameGraspPrediction"]
  modelFileNamePlacePrediction = params["modelFileNamePlacePrediction"]
  minPointsPerSegment = params["minPointsPerSegment"]
  
  # cost factors
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  insignificantCost = params["insignificantCost"]
  
  # regrasp planning
  temporaryPlacePosition = params["temporaryPlacePosition"]
  nGraspSamples = params["nGraspSamples"]
  halfAngleOfGraspFrictionCone = params["halfAngleOfGraspFrictionCone"]
  nTempPlaceSamples = params["nTempPlaceSamples"]
  nGoalPlaceSamples = params["nGoalPlaceSamples"]
  maxSearchDepth = params["maxSearchDepth"]
  maxIterations = params["maxIterations"]
  
  # visualization/saving
  showViewer = False
  showWarnings = False
  
  # INITIALIZATION =================================================================================
  
  # disable GPU usage to avoid out of memory error
  tensorflow.config.set_visible_devices([], "GPU")  
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveSensor()
  env.RemoveFloatingHand()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
  # initialize regrasp planner
  regraspPlanner = PlannerRegraspSP(env, temporaryPlacePosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples,
    modelFileNameGraspPrediction, modelFileNamePlacePrediction, taskCostFactor, regraspPlanStepCost,
    graspUncertCostFactor, placeUncertCostFactor, insignificantCost, maxSearchDepth, maxIterations,
    minPointsPerSegment)
  
  # RUN TEST =======================================================================================
  
  listener = Listener(("localhost", 7000 + wid))
  
  while True:
    
    print("Waiting for connection...")
    
    connection = listener.accept()
    print("Connection received.")
    
    message = connection.recv()
    print("Message received.")
    
    # message assumed to be a list: [purpose, data[0], ..., data[n]]
    purpose = message[0]
    data = message[1:]
    
    if purpose == "regrasp":
      
      print("Starting regrasp planner. ----------------")      
      
      # call plan
      pickPlaces, configs, goalIdx, cost = regraspPlanner.PlanWorker(data[0], data[1], data[2],
        data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
        connection)
      
      # assemble return message
      message = ["regrasp", pickPlaces, configs, goalIdx, cost]

    # send back result
    connection.send(message)

if __name__ == "__main__":
  
  try:
    wid = int(sys.argv[1])
  except:
    print("Usage: python/worker_blocks_sp.py wid")
    exit()  
  
  main(wid)