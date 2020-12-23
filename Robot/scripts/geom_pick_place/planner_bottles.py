'''TODO'''

# python
# scipy
from numpy.linalg import eig
from numpy.random import choice, uniform
from scipy.spatial import cKDTree
from numpy import arange, argmax, argmin, argsort, array, cov, cross, dot, eye, mean, ones, pi, \
  power, sqrt, sum, zeros
# openrave
import openravepy
# self
from hand_descriptor import HandDescriptor
from planner import Planner
import point_cloud

class PlannerBottles(Planner):

  def __init__(self, env, nGoalOrientations, preGraspOffset, minPointsPerSegment):
    '''TODO'''
    
    Planner.__init__(self, env, preGraspOffset, minPointsPerSegment)
    
    # parameters
    self.nGoalOrientations = nGoalOrientations
    
  def CompleteObjects(self, clouds, model, useGroundTruth):
    '''TODO'''
    
    if useGroundTruth:
      
      # preprocessing
      modelTrees = []
      for obj in self.env.unplacedObjects:
        bTo = dot(obj.GetTransform(), point_cloud.InverseTransform(obj.cloudTmodel))
        modelCloud = point_cloud.Transform(bTo, obj.cloud)
        modelTrees.append(cKDTree(modelCloud))
      
      completions = []; compCorrectProbs = []
      for cloud in clouds:
        
        # figure out which model corresponds to this segmentation
        distances = []
        for tree in modelTrees:
          d, _ = tree.query(cloud)
          distances.append(mean(d))
          
        # sample points on the full cloud for this model
        modelCloud = modelTrees[argmin(distances)].data
        idx = choice(arange(modelCloud.shape[0]), size = model.nInputPoints)
        completions.append(modelCloud[idx])
        compCorrectProbs.append(ones(model.nInputPoints))
        
      return completions, compCorrectProbs
    
    return super(PlannerBottles, self).CompleteObjects(clouds, model, False)
    
  def GetGoalPoses(self, bottleClouds, coasterClouds):
    '''TODO'''
    
    # 1. Error checking.

    goalPoses = []; goalCosts = []; targObjIdxs = []
    
    if len(bottleClouds) == 0:
      print("No bottles were detected!")
      return goalPoses, goalCosts, targObjIdxs
    if len(coasterClouds) == 0:
      print("No coasters were detected!")
      return goalPoses, goalCosts, targObjIdxs
      
    # 2. Decide which coaster to use.
        
    # use the coaster most distant from the robot base. the placed bottle will become an obstacle.
    coasterCenters = []
    for cloud in coasterClouds:
      coasterCenters.append(mean(cloud, axis = 0))
    coasterCenters = array(coasterCenters)
    targSupIdx = argmax(sum(power(coasterCenters, 2), axis = 1))
    
    coasterCloud = coasterClouds[targSupIdx]
    coasterCenter = coasterCenters[targSupIdx]
    
    # 3. Plan goals for each bottle.
    
    for bottleIdx, bottleCloud in enumerate(bottleClouds):
    
      # 4. Determine long axis of bottle.
      
      # compute covariance of points
      S = cov(bottleCloud, rowvar = False)
      
      # find direction with largest variance (eigenvector with largest eigenvalue)
      values, vectors = eig(S)
      longest = vectors[:, argmax(values)]
      #self.PlotLongestAxis(bottleCloud, longest, bottleCenter)
        
      # 5. Locate the bottom of the bottle.      
      
      # project bottom/top points on ends
      bottleCenter = mean(bottleCloud, axis = 0)
      bTo = self.GetObjectFrame(longest, bottleCenter)
      oTb = point_cloud.InverseTransform(bTo)
      oX = point_cloud.Transform(oTb, bottleCloud)
      
      topIdx = array([]); botIdx = array([]); i = 1
      while sum(topIdx) <= 2 or sum(botIdx) <= 2:
        topIdx = oX[:, 2] >= max(oX[:, 2]) - i * 0.005
        botIdx = oX[:, 2] <= min(oX[:, 2]) + i * 0.005
        i += 1
      
      # fit ellipse to ends and take the 1 with largest area as the bottom
      topS = cov(oX[topIdx, 0:2], rowvar = False)
      botS = cov(oX[botIdx, 0:2], rowvar = False)
      topValues, topVectors = eig(topS)
      botValues, botVectors = eig(botS)
      areaTop = pi * sqrt(topValues[0]) * sqrt(topValues[1])
      areaBot = pi * sqrt(botValues[0]) * sqrt(botValues[1])
      if areaTop > areaBot:
        longest = -longest
        bTo = self.GetObjectFrame(longest, bottleCenter)
        oTb = point_cloud.InverseTransform(bTo)
        oX = point_cloud.Transform(oTb, bottleCloud)
      
      # 6. Sample goal poses.
      
      theta = uniform(0, 2 * pi, self.nGoalOrientations)
      rotAxis = array([0.0, 0.0, 1.0])
      position = array([coasterCenter[0], coasterCenter[1], max(coasterCloud[:, 2]) - min(oX[:, 2])])
      goals = []
      
      for i in xrange(self.nGoalOrientations):
        pTo = self.GetRotatedObjectFrame(rotAxis, position, theta[i])
        goals.append(dot(pTo, oTb))
        #self.env.PlotCloud(point_cloud.Transform(goalPoses[-1], bottleCloud))
        
      # 7. Add to lists.
      goalPoses += goals; goalCosts += [0.0] * len(goals); targObjIdxs += [bottleIdx] * len(goals)
      
    return goalPoses, goalCosts, targObjIdxs
    
  def GetObjectFrame(self, axis, center):
    '''TODO'''
    
    T = eye(4)
    T[0:3, 2] = axis
    T[0:3, 0] = self.GetOrthogonalUnitVector(axis)
    T[0:3, 1] = cross(T[0:3, 2], T[0:3, 0])
    T[0:3, 3] = center
    return T
    
  def GetRotatedObjectFrame(self, axis, center, theta):
    '''TODO'''
    
    T = eye(4)
    T[0:3, 2] = axis
    R = openravepy.matrixFromAxisAngle(axis, theta)[0:3, 0:3]
    T[0:3, 0] = dot(R, self.GetOrthogonalUnitVector(axis))
    T[0:3, 1] = cross(T[0:3, 2], T[0:3, 0])
    T[0:3, 3] = center
    return T
      
  def PlotLongestAxis(self, cloud, axis, center):
    '''TODO'''
    
    T = self.GetObjectFrame(axis, center)
    self.env.PlotDescriptors([HandDescriptor(T)])
    self.env.PlotCloud(cloud)
    
    raw_input("Showing longest axis.")
    
  def SegmentObjects(self, cloud, modelSegmentation, workspace, scoreThresh, useGroundTruth, nObjects):
    '''TODO'''
    
    # perform segmentation
    
    segmentedClouds, segmentationProbs = super(PlannerBottles, self).SegmentObjects(
      cloud, modelSegmentation, workspace, scoreThresh, useGroundTruth)
      
    if len(segmentedClouds) < nObjects:
      print("Less than {} objects segmented: nothing to place.".format(nObjects))
      return [], [], [], [], []
      
    # determine which are bottles and which are coasters
    
    centers = zeros((len(segmentedClouds), 3))
    for i, cloud in enumerate(segmentedClouds):
      centers[i, :] = mean(cloud, axis = 0)
      
    idx = argsort(centers[:, 2])    
    coasters = [segmentedClouds[idx[i]] for i in xrange(nObjects)]
    coasterCenters = [centers[idx[i]] for i in xrange(nObjects)]
    bottles = [segmentedClouds[idx[i]] for i in xrange(nObjects, len(segmentedClouds))]
    bottleProbs = [segmentationProbs[idx[i]] for i in xrange(nObjects, len(segmentedClouds))]
    bottleCenters = [centers[idx[i]] for i in xrange(nObjects, len(segmentedClouds))]
    
    '''for coaster in coasters:
      self.env.PlotCloud(coaster)
      raw_input("coaster")
    for bottle in bottles:
      self.env.PlotCloud(bottle)
      raw_input("bottle")'''
      
    # determine which bottles are over coasters
    
    freeBottles = []; freeBottleProbs = []; placedBottles = []; freeCoasters = [];
    occupiedCoasters = []
    
    for i, bottle in enumerate(bottles):
      nPlacedBottles = len(placedBottles)
      for j, coaster in enumerate(coasters):
        if sum(power(bottleCenters[i][0:2] - coasterCenters[j][0:2], 2)) < \
          (self.env.supportObjectDiameterRange[0] / 2.0)**2:
            placedBottles.append(bottle)
            occupiedCoasters.append(coaster)
            break
      if len(placedBottles) == nPlacedBottles:
        freeBottles.append(bottle)
        freeBottleProbs.append(bottleProbs[i])
        
    for coaster in coasters:
      isOccupied = False
      for occupiedCoaster in occupiedCoasters:
        if coaster is occupiedCoaster:
          isOccupied = True; break
      if not isOccupied:
        freeCoasters.append(coaster)
        
    # return result
    
    return freeBottles, freeBottleProbs, placedBottles, freeCoasters, occupiedCoasters