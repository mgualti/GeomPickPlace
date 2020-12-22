#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import array, pi
  
def Parameters():
  '''Specifies simulation hyperparameters.'''
  
  # perception
  minPointsPerSegment = 1

  # cost factors
  taskCostFactor = 0.00
  regraspPlanStepCost = 1.00e-3
  segmUncertCostFactor = 1.00
  compUncertCostFactor = 1.00
  graspUncertCostFactor = 1.00
  placeUncertCostFactor = 1.00
  antipodalCostFactor = 0.00
  insignificantCost = 1.00e-6
  
  # task planning
  nInitialGraspSamples = 500
  loweringDelta = 0.01
  collisionCubeSize = 0.01
  nWorkers = 8
  maxGoalsPerWorker = 2
  maxTaskPlanningTime = 60.0
  
  # regrasp planning
  temporaryPlacePosition = array([0.00, 0.40, 0.00])
  nGraspSamples = 500
  halfAngleOfGraspFrictionCone = 12.0 * pi / 180
  nTempPlaceSamples = 3
  nGoalPlaceSamples = nWorkers * maxGoalsPerWorker
  maxSearchDepth = 6
  maxIterations = 20
  
  # return parameters
  return locals()