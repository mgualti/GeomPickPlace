#!/usr/bin/env python
'''TODO'''

# python
import sys
from multiprocessing.connection import Listener
# scipy
# self
from place_bottles_params_mc import Parameters
from geom_pick_place.planner_regrasp_mc import PlannerRegraspMC
from geom_pick_place.environment_bottles import EnvironmentBottles

# uncomment when profiling
#import os; os.chdir("/home/mgualti/GeomPickPlace/Simulation")

def main(wid):
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # perception
  minPointsPerSegment = params["minPointsPerSegment"]
  
  # cost factors
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  segmUncertCostFactor = params["segmUncertCostFactor"]
  compUncertCostFactor = params["compUncertCostFactor"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  antipodalCostFactor = params["antipodalCostFactor"]
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
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
  env.RemoveSensor()
  env.RemoveFloatingHand()
  temporaryPlacePosition[2] = env.GetTableHeight()
  
  # initialize regrasp planner
  regraspPlanner = PlannerRegraspMC(env, temporaryPlacePosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor,
    regraspPlanStepCost, segmUncertCostFactor, compUncertCostFactor, graspUncertCostFactor,
    placeUncertCostFactor, antipodalCostFactor, insignificantCost, maxSearchDepth, maxIterations,
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
        data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12],
        connection)
      
      # assemble return message
      message = ["regrasp", pickPlaces, configs, goalIdx, cost]

    # send back result
    connection.send(message)

if __name__ == "__main__":
  
  try:
    wid = int(sys.argv[1])
  except:
    print("Usage: python/worker_blocks.py wid")
    exit()  
  
  main(wid)