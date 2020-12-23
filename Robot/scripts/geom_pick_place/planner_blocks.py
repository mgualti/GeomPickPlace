'''TODO'''

# python
from copy import copy
# scipy
from numpy.linalg import norm
from scipy.stats import rankdata
from numpy import argmax, argmin, argsort, array, cos, cross, dot, eye, mean, pi
# self
from planner import Planner
import point_cloud

class PlannerBlocks(Planner):

  def __init__(self, env, preGraspOffset, minPointsPerSegment, blockRowStart):
    '''TODO'''
    
    Planner.__init__(self, env, preGraspOffset, minPointsPerSegment)
    
    # parameters
    
    self.blockRowStart = blockRowStart
    
    self.goalRotations = [\
      array([[ 1.0,  0.0, 0.0, 0.0], [ 0.0,  1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
      array([[ 0.0,  1.0, 0.0, 0.0], [-1.0,  0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
      array([[-1.0,  0.0, 0.0, 0.0], [ 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
      array([[ 0.0, -1.0, 0.0, 0.0], [ 1.0,  0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])]
  
  def GetGoalPoses(self, clouds, normals, blockRowCloud):
    '''TODO'''
    
    # parameters
    sameNormalThresh = cos(45 * pi / 180)
    
    # find the 6 faces of each box
    facePoints = []; faceNormals = []
    for i in xrange(len(clouds)):
      
      cloud = clouds[i]; normal = normals[i]
      
      normalBins = []; cloudBins = []; normalBinsCounts = []
      for j in xrange(cloud.shape[0]):
        didBin = False
        for k in xrange(len(normalBins)):
          averageNormal = normalBins[k] / normalBinsCounts[k]
          if dot(normal[j], averageNormal) >= norm(averageNormal) * sameNormalThresh:
            normalBins[k] += normal[j]
            cloudBins[k] += cloud[j]
            normalBinsCounts[k] += 1
            didBin = True
            break
        if not didBin:
          normalBins.append(copy(normal[j]))
          cloudBins.append(copy(cloud[j]))
          normalBinsCounts.append(1)          
      
      idxs = argsort(-array(normalBinsCounts))[:min(6, len(normalBinsCounts))]
      faceNormals.append(array([normalBins[j] / norm(normalBins[j]) for j in idxs]))
      facePoints.append(array([cloudBins[j] / normalBinsCounts[j] for j in idxs]))
    
    # find the longest dimension of each object
    maxFaceSep = []; maxSepFacePoints = []; maxSepFaceNormals = []
    for i in xrange(len(clouds)):    
      
      maxFaceSep.append(0); maxSepFacePoints.append(None); maxSepFaceNormals.append(None)      
      
      pFace = facePoints[i]; nFace = faceNormals[i]
      for j in xrange(pFace.shape[0]):
        
        for k in xrange(j + 1, pFace.shape[0]):
          if dot(nFace[j], -nFace[k]) >= sameNormalThresh:
            # faces are anti-parallel
            x = pFace[j] - pFace[k]
            n = nFace[j] - nFace[k]
            n = n / norm(n)
            d = abs(dot(n, x))
            if d > maxFaceSep[-1]:
              maxFaceSep[-1] = d
              maxSepFacePoints[-1] = array([pFace[j], pFace[k]])
              maxSepFaceNormals[-1] = array([nFace[j], nFace[k]])
    
    costs = rankdata(-array(maxFaceSep, dtype='float32'), method = "ordinal") - 1.0
    
    # determine goal poses
    
    goalPoses = []; goalCosts = []; targObjIdxs = []
    
    for i in xrange(len(clouds)):
      
      for j in xrange(2):
        
        # attach a reference frame to the point cloud, based on the desired up direction and orthogonal faces
        up = maxSepFaceNormals[i][j]
        mostOrthoIdx = argmin(abs(dot(faceNormals[i], up.T)))
        orthoNearest = self.GetNearestOrthogonalUnitVector(up, faceNormals[i][mostOrthoIdx, :])
        center = mean(clouds[i], axis = 0)
        
        bTo = eye(4)
        bTo[0:3, 0] = orthoNearest
        bTo[0:3, 1] = cross(up, orthoNearest)
        bTo[0:3, 2] = up
        bTo[0:3, 3] = center
        
        for k in xrange(4):
          
          # determine the desired, new object orientation
          bToo = copy(self.goalRotations[k])
          Tchange = dot(bToo, point_cloud.InverseTransform(bTo))
          cloudRotated = point_cloud.Transform(Tchange, clouds[i])
          
          # determine goal translation
          xOffset = min(blockRowCloud[:, 0]) - 0.01 if blockRowCloud.shape[0] > 0 \
            else self.blockRowStart[0]
          x = xOffset - max(cloudRotated[:, 0])
          y = self.blockRowStart[1]
          z = self.blockRowStart[2] - min(cloudRotated[:, 2])
          bToo[0:3, 3] = array([x, y, z])
          Tchange = dot(bToo, point_cloud.InverseTransform(bTo))
          
          goalPoses.append(Tchange)
          goalCosts.append(costs[i])
          targObjIdxs.append(i)
      
    return goalPoses, goalCosts, targObjIdxs