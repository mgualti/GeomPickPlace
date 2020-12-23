#!/usr/bin/env python
'''TODO'''

# python
import sys
import time
from multiprocessing.connection import Listener
# scipy
# self
from place_bottles_params import Parameters
from geom_pick_place.planner_regrasp import PlannerRegrasp
from geom_pick_place.environment_bottles import EnvironmentBottles

# uncomment when profiling
#import os; os.chdir("/home/mgualti/GeomPickPlace/Simulation")

def main(wid):
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  params = Parameters()
  
  # cost factors
  taskCostFactor = params["taskCostFactor"]
  regraspPlanStepCost = params["regraspPlanStepCost"]
  segmUncertCostFactor = params["segmUncertCostFactor"]
  compUncertCostFactor = params["compUncertCostFactor"]
  graspUncertCostFactor = params["graspUncertCostFactor"]
  placeUncertCostFactor = params["placeUncertCostFactor"]
  insignificantCost = params["insignificantCost"]
  
  # general planning  
  preGraspOffset = params["preGraspOffset"]
  minPointsPerSegment = params["minPointsPerSegment"]
  
  # regrasp planning
  regraspPosition = params["regraspPosition"]
  nGraspSamples = params["nGraspSamples"]
  graspFrictionCone = params["graspFrictionCone"]
  graspContactWidth = params["graspContactWidth"]
  nTempPlaceSamples = params["nTempPlaceSamples"]
  nGoalPlaceSamples = params["nGoalPlaceSamples"]
  maxSearchDepth = params["maxSearchDepth"]
  maxTimeToImprovePlan = params["maxTimeToImprovePlan"]
  maxTimeToPlan = params["maxTimeToPlan"]
  
  # visualization/saving
  showViewer = False
  showWarnings = False
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBottles(showViewer, showWarnings)
  env.RemoveFloatingHand()
  env.RemoveSensor()
  regraspPosition[2] = env.GetTableHeight()
  
  # initialize regrasp planner
  regraspPlanner = PlannerRegrasp(env, preGraspOffset, minPointsPerSegment, regraspPosition,
    nGraspSamples, graspFrictionCone, graspContactWidth, nTempPlaceSamples, nGoalPlaceSamples,
    taskCostFactor, regraspPlanStepCost, segmUncertCostFactor, compUncertCostFactor,
    graspUncertCostFactor, placeUncertCostFactor, insignificantCost, maxSearchDepth,
    maxTimeToImprovePlan, maxTimeToPlan)
  
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
      pickPlaces, goalIdx, cost = regraspPlanner.PlanWorker(data[0], data[1], data[2],
        data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
        connection)
      
      # assemble return message
      message = ["regrasp", pickPlaces, goalIdx, cost]

    # send back result
    connection.send(message)
    time.sleep(1)
    connection.close()

if __name__ == "__main__":
  
  try:
    wid = int(sys.argv[1])
  except:
    print("Usage: python/worker_blocks.py wid")
    exit()  
  
  main(wid)