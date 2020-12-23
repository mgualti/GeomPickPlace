#!/usr/bin/env python
'''TODO'''

# python
import sys
import time
from multiprocessing.connection import Listener
# scipy
# tensorflow
import tensorflow
# self
from place_packing_sp_params import Parameters
from geom_pick_place.planner_packing import PlannerPacking
from geom_pick_place.planner_regrasp_sp import PlannerRegraspSP
from geom_pick_place.environment_packing import EnvironmentPacking

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
  
  # general planning
  preGraspOffset = params["preGraspOffset"]  
  
  # task planning
  nInitialGraspSamples = params["nInitialGraspSamples"]
  loweringDelta = params["loweringDelta"]
  collisionCubeSize = params["collisionCubeSize"]
  nWorkers = params["nWorkers"]
  maxGoalsPerWorker = params["maxGoalsPerWorker"]
  maxTaskPlanningTime = params["maxTaskPlanningTime"]
  
  # regrasp planning
  regraspPosition = params["regraspPosition"]
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
  env = EnvironmentPacking(showViewer, showWarnings)
  env.RemoveSensor()
  env.RemoveFloatingHand()
  regraspPosition[2] = env.GetTableHeight()
  
  # initialize task planner
  taskPlanner = PlannerPacking(env, nInitialGraspSamples, loweringDelta, collisionCubeSize,
    nWorkers, maxGoalsPerWorker, maxTaskPlanningTime, minPointsPerSegment, preGraspOffset) 
  
  # initialize regrasp planner
  regraspPlanner = PlannerRegraspSP(env, regraspPosition, nGraspSamples,
    halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples,
    modelFileNameGraspPrediction, modelFileNamePlacePrediction, taskCostFactor, regraspPlanStepCost,
    graspUncertCostFactor, placeUncertCostFactor, insignificantCost, maxSearchDepth, maxIterations,
    minPointsPerSegment, preGraspOffset)
  
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
      
    elif purpose == "regrasp-continue":
      
      print("Continuing regrasp planner. ----------------")
      
      # call plan
      pickPlaces, goalIdx, cost = regraspPlanner.ContinueWorker(connection)
      
      # assemble return message
      message = ["regrasp-continue", pickPlaces, goalIdx, cost]
      
    elif purpose == "packing":
      
      print("Starting packing planner. ----------------")
      
      # call plan
      goalPoses, goalCosts, targObjIdxs = taskPlanner.GetGoalPosesWorker(
        data[0], data[1], data[2], data[3])
      
      # assemble return message
      message = ["packing", goalPoses, goalCosts, targObjIdxs]

    # send back result
    connection.send(message)
    time.sleep(1)
    connection.close()

if __name__ == "__main__":
  
  try:
    wid = int(sys.argv[1])
  except:
    print("Usage: python/worker_regrasp.py wid")
    exit()  
  
  main(wid)