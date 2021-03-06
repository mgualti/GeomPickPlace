#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import array, pi
  
def Parameters():
  '''Specifies simulation hyperparameters.'''

  # perception
  modelFileNameGraspPrediction = "pcn_model_grasp_prediction_packing.h5"
  modelFileNamePlacePrediction = "pcn_model_place_prediction_packing.h5"
  minPointsPerSegment = 1

  # cost factors
  taskCostFactor = 0.00
  regraspPlanStepCost = 0.00
  graspUncertCostFactor = 1.00
  placeUncertCostFactor = 1.00
  insignificantCost = 1.00e-6
  
  # task planning
  maxGoalsPerWorker = 1
  
  # regrasp planning
  temporaryPlacePosition = array([0.00, 0.40, 0.00])
  nGraspSamples = 500
  halfAngleOfGraspFrictionCone = 12.0 * pi / 180
  nTempPlaceSamples = 3
  nGoalPlaceSamples = maxGoalsPerWorker
  maxSearchDepth = 6
  maxIterations = 20
  
  # return parameters
  return locals()