#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import array, pi
  
def Parameters():
  '''Specifies simulation hyperparameters.'''
  
  # perception
  modelFileNameGraspPrediction = "pcn_model_grasp_prediction_blocks.h5"
  modelFileNamePlacePrediction = "pcn_model_place_prediction_blocks.h5"
  minPointsPerSegment = 1  
  
  # cost factors
  taskCostFactor = 0.00
  regraspPlanStepCost = 0.00
  graspUncertCostFactor = 1.00
  placeUncertCostFactor = 1.00
  insignificantCost = 1.00e-6
  
  # regrasp planning
  temporaryPlacePosition = array([0.00, 0.40, 0.00])
  nGraspSamples = 100
  halfAngleOfGraspFrictionCone = 5.0 * pi / 180
  nTempPlaceSamples = 1
  nGoalPlaceSamples = 1
  maxSearchDepth = 6
  maxIterations = 30
  
  # return parameters
  return locals()