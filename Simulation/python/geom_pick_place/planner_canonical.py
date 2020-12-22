'''TODO'''

# python
# scipy
from numpy import argmin, dot, eye, mean, power, reshape, sum, tile
# openrave
# self
import point_cloud
from planner import Planner

class PlannerCanonical(Planner):

  def __init__(self, env, nGoalsPerObject, minPointsPerSegment):
    '''TODO'''
    
    Planner.__init__(self, env, minPointsPerSegment)
    
    # parameters
    self.nGoalsPerObject = nGoalsPerObject
    
  def GetGoalPoses(self, clouds, goalPosition):
    '''TODO'''
    
    # 1. For each cloud (estimated object), find the nearest actual object.

    objects = self.env.unplacedObjects
    actualCenters = self.env.GetObjectCentroids(objects)
    
    goalPoses = []; goalCosts = []; targObjIdxs = []
    for i in xrange(len(clouds)):
      
      estimatedCenter = mean(clouds[i], axis = 0)
      nearestObjIdx = argmin(sum(power(actualCenters - tile(reshape(estimatedCenter, (1, 3)),
        (actualCenters.shape[0], 1)), 2), axis = 1))
      nearestObj = objects[nearestObjIdx]
      
      # 2. Find canonical pose of object and transform needed for achieving that pose.
      bTg = eye(4)
      bTg[0:2, 3] = goalPosition
      bTg[2, 3] = self.env.GetTableHeight() - min(nearestObj.cloud[:, 2])
      bTs = nearestObj.GetTransform()
      sTb = point_cloud.InverseTransform(bTs)
      X = dot(bTg, sTb) # solution to bTg = X bTs
      
      # visualize placement
      #cloudAtStart = point_cloud.Transform(bTs, nearestObj.cloud)
      #self.env.PlotCloud(point_cloud.Transform(X, cloudAtStart))
      
      # 3. Save result.
      goalPoses.append(X)
      goalCosts.append(0.0)
      targObjIdxs.append(i)
    
    # 4. Return solutions.
    return goalPoses, goalCosts, targObjIdxs