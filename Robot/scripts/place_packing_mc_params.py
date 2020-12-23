#!/usr/bin/env python
'''TODO'''

# scipy
from numpy import pi
# point_cloud
import point_cloud
  
def Parameters():
  '''Specifies simulation hyperparameters.'''

  # perception
  minPointsPerSegment = 10

  # cost factors
  taskCostFactor = 0.00
  regraspPlanStepCost = 1.00e-3
  segmUncertCostFactor = 1.00
  compUncertCostFactor = 1.00
  graspUncertCostFactor = 1.00
  placeUncertCostFactor = 1.00
  antipodalCostFactor = 0.00
  insignificantCost = 1.00e-6
  
  # general planning
  preGraspOffset = 0.055
  
  # task planning
  nInitialGraspSamples = 500
  loweringDelta = 0.01
  collisionCubeSize = 0.01
  nWorkers = 8
  maxGoalsPerWorker = 2
  maxTaskPlanningTime = 60.0
  
  # regrasp planning
  regraspWorkspace = [[0.45 - 0.12, 0.45 + 0.12], [0.29 - 0.12, 0.29 + 0.12], [0.01, 1.00]]
  regraspPosition = point_cloud.WorkspaceCenter(regraspWorkspace)
  nGraspSamples = 500
  halfAngleOfGraspFrictionCone = 12.0 * pi / 180
  nTempPlaceSamples = 3
  nGoalPlaceSamples = nWorkers * maxGoalsPerWorker
  maxSearchDepth = 6
  maxIterations = 15
  
  # return parameters
  return locals()