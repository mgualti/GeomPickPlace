'''TODO'''

# python
import os
import fnmatch
from copy import copy
# scipy
from scipy.io import loadmat
from scipy.spatial import cKDTree
from numpy.random import normal, randint, uniform
from numpy import argmax, array, cos, dot, ones, pi
# openrave
import openravepy
# self
import point_cloud
from environment_pick_place import EnvironmentPickPlace

class EnvironmentBlocks(EnvironmentPickPlace):

  def __init__(self, showViewer, showWarnings):
    '''TODO'''
    
    EnvironmentPickPlace.__init__(self, showViewer, showWarnings)
    
    # parameters (grasp simulation)
    self.halfAngleOfGraspFrictionCone = 5.0 * pi / 180.0 # radians
    self.cosHalfAngleOfGraspFrictionCone = cos(self.halfAngleOfGraspFrictionCone)
    self.graspContactWidth = 0.001 # meters
    
    # parameters (place simulation)
    self.faceDistTol = 0.004
    
  def GenerateRandomBlock(self, extents, name):
    '''TODO'''

    extents = [uniform(extents[0][0], extents[0][1]),
               uniform(extents[1][0], extents[1][1]),
               uniform(extents[2][0], extents[2][1])]
   
    return self.GenerateKinBody(extents, name)
    
  def EvaluateArrangement(self):
    '''TODO'''
    
    nPlaced = len(self.placedObjects)
    
    lastHeight = float('inf')
    longestEndUp = []; orderCorrect = []
    for obj in self.placedObjects:
      
      # determine side lengths
      lengths = array([
        max(obj.cloud[:, 0]) - min(obj.cloud[:, 0]),
        max(obj.cloud[:, 1]) - min(obj.cloud[:, 1]),
        max(obj.cloud[:, 2]) - min(obj.cloud[:, 2]) ])
      longestSide = argmax(lengths)
      
      # determine which side is up
      T = obj.GetTransform()
      z = array([0, 0, 1])
      up = argmax([abs(dot(T[0:3, 0], z)), abs(dot(T[0:3, 1], z)), abs(dot(T[0:3, 2], z))])
      
      # determine ordering and longest side up
      orderCorrect.append(lastHeight > lengths[longestSide])
      longestEndUp.append(longestSide == up)
      lastHeight = lengths[longestSide]
    
    return nPlaced, orderCorrect, longestEndUp
    
  def ExecuteRegraspPlan(self, pickPlaces, plannedConfigs, targObjCloud, showSteps):
    '''TODO'''
    
    prnt = lambda s: self.PrintString(s, showSteps)
    picks = [pickPlaces[i] for i in xrange(0, len(pickPlaces), 2)]
    places = [pickPlaces[i] for i in xrange(1, len(pickPlaces), 2)]
    
    isTempPlaceStable = []; isGraspSuccess = []
    
    for i, pick in enumerate(picks):
      
      isGraspSuccess.append(False)
      
      # figure out which object is being grasped
      objsInHand = self.FindObjectsInHand(pick)
      if len(objsInHand) == 0:
        prnt("Nothing in the hand!")
        self.RemoveObjectNearestCloud(targObjCloud)
        return False, isGraspSuccess, isTempPlaceStable
      if len(objsInHand) > 1:
        prnt("Multiple objects in the hand!")
        self.RemoveObjectNearestCloud(targObjCloud)
        return False, isGraspSuccess, isTempPlaceStable
      objInHand = objsInHand[0]
      
      # move arm to grasp pose
      self.MoveRobot(plannedConfigs[2 * i])
      prnt("Grasp {} / {}.".format(i + 1, len(picks)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at grasp.")
        self.RemoveUnplacedObject(objInHand)
        return False, isGraspSuccess, isTempPlaceStable, 
        
      # check if grasp is antipodal on underlying object
      cloud, normals = point_cloud.Transform( \
        objInHand.GetTransform(), objInHand.cloud, objInHand.normals)
      isAntipodal, _ = self.IsAntipodalGrasp(pick, cloud, normals,
        self.cosHalfAngleOfGraspFrictionCone, self.graspContactWidth)
      if not isAntipodal:
          prnt("Grasp is not antipodal.")
          self.RemoveUnplacedObject(objInHand)
          return False, isGraspSuccess, isTempPlaceStable
      isGraspSuccess[-1] = True
        
      # move arm to place pose
      self.MoveRobot(plannedConfigs[2 * i + 1])
      self.MoveObjectToHandAtGrasp(pick, objInHand)
      prnt("Moved object to place {} / {}".format(i + 1, len(places)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at place.")
        self.RemoveUnplacedObject(objInHand)
        return False, isGraspSuccess, isTempPlaceStable
        
      # if this is a temporary place, check if the object is resting stably
      if i != len(picks) - 1:
        cloud = point_cloud.Transform(objInHand.GetTransform(), objInHand.cloud)
        isStable = self.IsPlacementStable(cloud, self.faceDistTol)
        isStableString = "is" if isStable else "is not"
        print("Temporary place {} stable.".format(isStableString))
        isTempPlaceStable.append(isStable)
        
    # indicate this object as being placed
    self.placedObjects.append(objInHand)
    self.unplacedObjects.remove(objInHand)
    
    return True, isGraspSuccess, isTempPlaceStable
    
  def LoadInitialScene(self, nObjects, cloudDirectory, workspace, maxPlaceAttempts = 30):
    '''TODO'''
    
    self.ResetScene()
    cloudFileNames = os.listdir(cloudDirectory)
    cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
    
    for i in xrange(nObjects):
      
      # choose a random object from the folder
      fileName = cloudFileNames[randint(len(cloudFileNames))]
      
      # load object
      body = self.LoadObjectFromFullCloudFile(cloudDirectory, fileName, "object-{}".format(i))
      
      # select pose for object
      for j in xrange(maxPlaceAttempts):
        # choose orientation
        self.RandomizeObjectPose(body, workspace)
          
        if not self.env.CheckCollision(body):
          break
          
  def LoadObjectFromFullCloudFile(self, cloudDirectory, cloudFileName, name):
    '''TODO'''
    
    # load cloud data
    data = loadmat(cloudDirectory + "/" + cloudFileName)
    
    # generate mesh
    body = self.GenerateKinBody(data["extents"].flatten(), name)
    body.cloud = data["cloud"]
    body.normals = data["normals"]
    self.unplacedObjects.append(body)
    
    return body
    
  def RandomizeObjectPose(self, body, workspace):
    '''TODO'''
    
    # choose an up axis
    upChoices = [array([[0, 0, -1, 0], [0, 1,  0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                 array([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
                 array([[1, 0,  0, 0], [0, 1,  0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]
    upIdx = randint(3)
    R1 = upChoices[upIdx]
    
    # choose a rotation about the up axis
    r2 = uniform(0, 2 * pi)
    R2 = openravepy.matrixFromAxisAngle([0.0, 0.0, 1.0], r2)
    
    # compute orientation
    T = dot(R2, R1) # about fixed frame in order 1, 2
    
    # choose xy position
    T[0:2, 3] = array([ \
      uniform(workspace[0][0], workspace[0][1]),
      uniform(workspace[1][0], workspace[1][1])])
      
    # choose height
    T[2, 3] = self.GetTableHeight() + copy(body.extents[upIdx]) / 2.0
    
    # set transform
    body.SetTransform(T)
    
    # return transform    
    return T