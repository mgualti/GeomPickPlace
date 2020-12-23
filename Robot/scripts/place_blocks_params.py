#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import pi
# point_cloud
import point_cloud
  
def Parameters():
  '''Specifies simulation hyperparameters.'''
  
  # cost factors
  taskCostFactor = 1.00
  regraspPlanStepCost = 0.10
  segmUncertCostFactor = 0.00
  compUncertCostFactor = 0.00
  graspUncertCostFactor = 0.00
  placeUncertCostFactor = 0.00
  insignificantCost = 1.00e-2
  
  # general planning
  preGraspOffset = 0.05
  minPointsPerSegment = 30
  
  # regrasp planning
  regraspWorkspace = [[0.22, 0.34], [0.24, 0.36], [0.01, 1.00]]
  regraspPosition = point_cloud.WorkspaceCenter(regraspWorkspace)
  nGraspSamples = 100
  graspFrictionCone = 3.0 * pi / 180
  graspContactWidth = 0.005
  nTempPlaceSamples = 1
  nGoalPlaceSamples = 1
  maxSearchDepth = 6
  maxTimeToImprovePlan = 30.0
  maxTimeToPlan = 60.0
  
  # return parameters
  return locals()