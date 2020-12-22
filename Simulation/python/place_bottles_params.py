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
  regraspPlanStepCost = 0.00
  segmUncertCostFactor = 0.00
  compUncertCostFactor = 0.00
  graspUncertCostFactor = 0.00
  placeUncertCostFactor = 0.00
  antipodalCostFactor = 0.00
  insignificantCost = 1.00e-6
  
  # regrasp planning
  temporaryPlacePosition = array([0.00, 0.40, 0.00])
  nGraspSamples = 100
  halfAngleOfGraspFrictionCone = 12.0 * pi / 180
  nTempPlaceSamples = 1
  nGoalPlaceSamples = 1
  maxSearchDepth = 6
  maxIterations = 30
  
  # return parameters
  return locals()