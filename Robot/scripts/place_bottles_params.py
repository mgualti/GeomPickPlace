#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import pi
# self
import point_cloud
  
def Parameters():
  '''Specifies simulation hyperparameters.'''
  
  # cost factors
  taskCostFactor = 1.00
  regraspPlanStepCost = 1.00
  segmUncertCostFactor = 0.00
  compUncertCostFactor = 0.00
  graspUncertCostFactor = 0.00
  placeUncertCostFactor = 0.00
  insignificantCost = 1.00e-3
  
  # general planning
  preGraspOffset = 0.070
  minPointsPerSegment = 30
  
  # regrasp planning
  regraspWorkspace = [[0.490 - 0.110, 0.490 + 0.110], [-0.285 - 0.110, -0.285 + 0.110], [0.01, 1.00]]
  regraspPosition = point_cloud.WorkspaceCenter(regraspWorkspace)
  nGraspSamples = 100
  graspFrictionCone = 5.0 * pi / 180
  graspContactWidth = 0.005
  nTempPlaceSamples = 1
  nGoalPlaceSamples = 1
  maxSearchDepth = 6
  maxTimeToImprovePlan = 30.0
  maxTimeToPlan = 40.0
  
  # return parameters
  return locals()