'''TODO'''

# python
import os
import re
import fnmatch
# scipy
from scipy.io import loadmat
from numpy.linalg import eig
from scipy.linalg import norm
from scipy.spatial import cKDTree
from numpy.random import choice, normal, randint, uniform
from numpy import arange, argmax, argmin, array, cos, cov, dot, empty, eye, hstack, maximum, mean, \
  pi, power, reshape, sin, sum, tile, zeros
# openrave
import openravepy
# self
import point_cloud
from environment_pick_place import EnvironmentPickPlace

class EnvironmentBottles(EnvironmentPickPlace):

  def __init__(self, showViewer, showWarnings):
    '''TODO'''
    
    EnvironmentPickPlace.__init__(self, showViewer, showWarnings)
    
    # parameters (object geometry)
    self.supportObjectHeightRange = [0.005, 0.020] # meters
    self.supportObjectDiameterRange = [0.08, 0.12] # meters
    
    # parameters (grasp simulation)
    self.halfAngleOfGraspFrictionCone = 15.0 * pi / 180.0 # radians
    self.cosHalfAngleOfGraspFrictionCone = cos(self.halfAngleOfGraspFrictionCone)
    self.graspContactWidth = 0.001 # meters
    
    # parameters (place simulation)
    self.placeOrientTolerance = 20.0 * pi / 180.0 # radians
    self.placeHeightTolerance = [0.02, 0.02] # meters
    self.faceDistTol = 0.004 # meters
    
    # initialization
    self.bottleObjects = []
    self.supportObjects = []
    self.occupiedSupports = []
    
  def AddSensorNoise(self, cloud, sigmaRotation, nNeighborhoods, sigmaNeighborhoodRadius,
    sigmaNeighborhoodZ, sigmaPointwiseZ):
    '''TODO'''
    
    # input checking
    if cloud.shape[0] == 0:
      return cloud
    
    # rotate the cloud
    axis = normal(size = 3)
    axis = axis / norm(axis)
    angle = normal(scale = sigmaRotation)
    T = openravepy.matrixFromAxisAngle(axis, angle)
    cloud = point_cloud.Transform(T, cloud)
    
    # neighborhood z-error
    neighborhoodSizes = normal(scale = sigmaNeighborhoodRadius, size = nNeighborhoods)
    neighborhoodSizes = maximum(zeros(nNeighborhoods), neighborhoodSizes)
    neighborhoodShifts = normal(scale = sigmaNeighborhoodZ, size = nNeighborhoods)
    neighborhoodLocations = cloud[choice(arange(cloud.shape[0]), nNeighborhoods)]
    
    tree = cKDTree(cloud)
    for i, location in enumerate(neighborhoodLocations):
      idx = tree.query_ball_point(location, neighborhoodSizes[i])
      cloud[idx, 2] += neighborhoodShifts[i]
      
    # point-wise z-error
    cloud[:, 2] += normal(scale = sigmaPointwiseZ, size = cloud.shape[0])
    
    # voxelize
    cloud = point_cloud.Voxelize(0.003, cloud)
    
    return cloud
    
  def EvaluateArrangement(self):
    '''TODO'''
    
    return len(self.placedObjects)
    
  def EvaluatePlacement(self, bottle):
    '''TODO'''
    
    bTo = dot(bottle.GetTransform(), point_cloud.InverseTransform(bottle.cloudTmodel))    
    
    # check if bottle is over a coaster    
    support = None
    for coaster in self.supportObjects:
      coasterXY = coaster.GetTransform()[0:2, 3]
      if sum(power(coasterXY - bTo[0:2, 3], 2)) < (coaster.extents[0] / 2.0)**2:
        support = coaster
        break
      
    # not above any coaster
    if support is None:
      return False, None
    
    # support object is already occupied
    if support in self.occupiedSupports:
      return False, None
   
    # check if bottle is vertical
    if not self.IsBottleUpright(bottle):
      return False, None
    
    # check if bottle bottom is within given height tolerance
    supportTopZ = support.GetTransform()[2, 3] + support.extents[1] / 2.0
    objectBottomZ = min(point_cloud.Transform(bTo, bottle.cloud)[:, 2])
    if objectBottomZ < supportTopZ - self.placeHeightTolerance[0] or \
       objectBottomZ > supportTopZ + self.placeHeightTolerance[1]:
         return False, None
    
    # placement is correct
    return True, support
    
  def ExecuteRegraspPlan(self, pickPlaces, plannedConfigs, targObjCloud, showSteps):
    '''TODO'''
    
    prnt = lambda s: self.PrintString(s, showSteps)
    picks = [pickPlaces[i] for i in xrange(0, len(pickPlaces), 2)]
    places = [pickPlaces[i] for i in xrange(1, len(pickPlaces), 2)]
    
    isTempPlaceStable = []; isGraspSuccess = []
    
    for i, pick in enumerate(picks):
      
      isGraspSuccess.append(False)
      
      # figure out which object is being grasped.
      objsInHand = self.FindObjectsInHand(pick)
      if len(objsInHand) == 0:
        prnt("Nothing in the hand!")
        self.RemoveBottleNearestCloud(targObjCloud)
        return isGraspSuccess, isTempPlaceStable
      if len(objsInHand) > 1:
        prnt("Multiple objects in the hand!")
        self.RemoveBottleNearestCloud(targObjCloud)
        return isGraspSuccess, isTempPlaceStable
      objInHand = objsInHand[0]
      if objInHand not in self.bottleObjects:
        prnt("The object in the hand is not a bottle.")
        self.RemoveBottleNearestCloud(targObjCloud)
        return isGraspSuccess, isTempPlaceStable
      
      # move arm to grasp pose
      self.MoveRobot(plannedConfigs[2 * i])
      prnt("Grasp {} / {}.".format(i + 1, len(picks)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at grasp.")
        self.RemoveBottle(objInHand)
        return isGraspSuccess, isTempPlaceStable
        
      # check if grasp is antipodal on underlying object
      cloud, normals = point_cloud.Transform(dot(objInHand.GetTransform(), \
        point_cloud.InverseTransform(objInHand.cloudTmodel)), objInHand.cloud, objInHand.normals)
      isAntipodal, _ = self.IsAntipodalGrasp(pick, cloud, normals,
        self.cosHalfAngleOfGraspFrictionCone, self.graspContactWidth)
      if not isAntipodal:
          prnt("Grasp is not antipodal.")
          self.RemoveBottle(objInHand)
          return isGraspSuccess, isTempPlaceStable
      isGraspSuccess[-1] = True
      
      # move arm to place pose
      self.MoveRobot(plannedConfigs[2 * i + 1])
      self.MoveObjectToHandAtGrasp(pick.T, objInHand)
      prnt("Moved object to place {} / {}".format(i + 1, len(places)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at place.")
        self.RemoveBottle(objInHand)
        return isGraspSuccess, isTempPlaceStable
        
      # if this is a temporary place, check if the object is resting stably
      if i != len(picks) - 1:
        cloud = point_cloud.Transform(dot(objInHand.GetTransform(), \
          point_cloud.InverseTransform(objInHand.cloudTmodel)), objInHand.cloud)
        isStable = self.IsPlacementStable(cloud, self.faceDistTol)
        isStableString = "is" if isStable else "is not"
        prnt("Temporary place {} stable.".format(isStableString))
        isTempPlaceStable.append(isStable)
    
    # evaluate placement: if bad, remove object, and, if good, update internal structures.
    placementCorrect, support = self.EvaluatePlacement(objInHand)
    
    # (if object was already placed, first update its status as being unplaced)
    if objInHand in self.placedObjects:
      idx = self.placedObjects.index(objInHand)
      self.placedObjects.pop(idx)
      self.occupiedSupports.pop(idx)
      self.unplacedObjects.append(objInHand)
      prnt("Object was already placed")
    
    # (if the placement is bad, remove the object and return)
    if not placementCorrect:
      prnt("Placement incorrect.")
      self.RemoveBottle(objInHand)
      return isGraspSuccess, isTempPlaceStable
    
    # (otherwise, the placement is good: add it to the list of placed objects)
    self.placedObjects.append(objInHand)
    self.occupiedSupports.append(support)
    self.unplacedObjects.remove(objInHand)
    return isGraspSuccess, isTempPlaceStable
    
  def FindObjectsInHand(self, bTh):
    '''Returns a list of objects intersecting the hand's rectangular closing region.
    - Input bTh: 4x4 numpy matrix, assumed to be a homogeneous transform, indicating the pose of the
      hand in the base frame.
    - Returns objectsInHand: Handles (of type KinBody) to objects in the hand.
    '''
    
    objectsInHand = []
    for i, obj in enumerate(self.objects):
      bTo = dot(obj.GetTransform(), point_cloud.InverseTransform(obj.cloudTmodel))
      hTo = dot(point_cloud.InverseTransform(bTh.T), bTo)
      X = point_cloud.Transform(hTo, obj.cloud)
      X = point_cloud.FilterWorkspace(self.handClosingRegion, X)
      if X.size > 0: objectsInHand.append(obj)
    return objectsInHand
    
  def GetSegmentation(self, cloud, nCategories):
    '''TODO'''
    
    # construct a KD-tree with the full cloud of each object
    fullCloudTrees = []
    for obj in self.objects:
      bTo = dot(obj.GetTransform(), point_cloud.InverseTransform(obj.cloudTmodel))
      objCloud = point_cloud.Transform(bTo, obj.cloud)
      fullCloudTrees.append(cKDTree(objCloud))
    
    # find the minimum distance between cloud points and object cloud points
    distances = []
    for i, obj in enumerate(self.objects):
      d, _ = fullCloudTrees[i].query(cloud)
      distances.append(reshape(d, (cloud.shape[0], 1)))
    distances = hstack(distances)
    
    # classify points based on the nearest object point
    segmentation = zeros((cloud.shape[0], nCategories), dtype = 'bool')
    rowIdx = arange(cloud.shape[0])
    colIdx = argmin(distances, axis = 1)    
    segmentation[rowIdx, colIdx] = 1
    
    return segmentation
    
  def IsBottleUpright(self, bottle):
    '''TODO'''
    
    T = bottle.GetTransform()
    return dot(array([0.0, 0.0, 1.0]), T[0:3, 1]) >= cos(self.placeOrientTolerance)
    
  def IsObjectWidthInLimits(self, cloud, minWidth = 0.030, maxWidth = 0.080):
    '''Project the object onto the x,y plane and and check that the widths are within given limits.
    - Input cloud: Point cloud of object (nx3 numpy array), assumed to be oriented upright.
    - Input minWidth: If smallest part is narrower than this (in meters), False is returned.
    - Input maxWidth: If largest part is greater than this (in meters), Fasle is returned.
    - Returns: True if the object width is in the limits and False otherwise.
    '''
    
    X = cloud[:, 0:2]
    S = cov(X, rowvar = False)
    values, vectors = eig(S)
    widestAxisIdx = argmax(values)
    widestAxis = vectors[:, widestAxisIdx]
    narrowestAxis = vectors[:, widestAxisIdx - 1]
    
    # rotate points so that least variance is in the x direction
    R = hstack([reshape(widestAxis, (2, 1)), reshape(narrowestAxis, (2, 1))])
    X = dot(R, X.T).T
    #V = hstack([X, zeros((X.shape[0], 1))])
    #point_cloud.Plot(V)
    
    widest = max(X[:, 0]) - min(X[:, 0])
    narrow = max(X[:, 1]) - min(X[:, 1])
    
    return narrow >= minWidth and widest <= maxWidth
    
  def LoadInitialScene(self, nObjects, cloudDirectoryBottles, cloudDirectoryCoasters, workspace,
    maxPlaceAttempts = 30):
    '''TODO'''
    
    # reset the initial scene    
    self.ResetScene()
    self.MoveRobotToHome()
    
    # load support objects
    cloudFileNames = os.listdir(cloudDirectoryCoasters)
    cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
    
    for i in xrange(nObjects):
      
      # choose a random object from the folder
      fileName = cloudFileNames[randint(len(cloudFileNames))]
      
      # load object
      body = self.LoadSupportObjectFromFullCloudFile(
        cloudDirectoryCoasters, fileName, "support-{}".format(i))
      
      # attempt to place support object with random, collision free position
      for j in xrange(maxPlaceAttempts):
        T = eye(4)
        T[0:2, 3] = array([
          uniform(workspace[0][0], workspace[0][1]),
          uniform(workspace[1][0], workspace[1][1])])
        T[2, 3] = body.extents[1] / 2.0 + self.GetTableHeight() + 0.001
        body.SetTransform(T)
        if not self.env.CheckCollision(body):
          break
        
    # load bottles
    cloudFileNames = os.listdir(cloudDirectoryBottles)
    cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
    
    for i in xrange(nObjects):
      
      # choose a random object from the folder
      fileName = cloudFileNames[randint(len(cloudFileNames))]
      
      # load object
      body = self.LoadObjectFromFullCloudFile(
        cloudDirectoryBottles, fileName, "object-{}".format(i))
      
      # select pose for object
      for j in xrange(maxPlaceAttempts):
        self.RandomizeObjectPose(body, workspace)
        if not self.env.CheckCollision(body): break
    
  def LoadObjectFromFullCloudFile(self, cloudDirectory, cloudFileName, name):
    '''TODO'''
    
    # load cloud data
    cloudData = loadmat(cloudDirectory + "/" + cloudFileName)
    height = float(cloudData["height"])
    scale = float(cloudData["scale"])
    cloud = cloudData["cloud"]
    normals = cloudData["normals"]
    cloudTmodel = cloudData["cloudTmodel"]
    meshFileName = cloudData["meshFileName"][0].encode("ascii")
    
    # load mesh
    self.env.Load(meshFileName, {"scalegeometry":str(scale)})
    body = self.env.GetKinBody(re.findall("/[^/]*.obj$", meshFileName)[0][1:-4])
    body.SetName(name)
    body.SetTransform(cloudTmodel)
    body.height = height
    body.scale = scale
    body.cloud = cloud
    body.normals = normals
    body.cloudTmodel = cloudTmodel
    body.meshFileName = meshFileName
    
    # add to internal variables
    self.objects.append(body)
    self.bottleObjects.append(body)
    self.unplacedObjects.append(body)
    
    return body
    
  def LoadSupportObjectFromFullCloudFile(self, cloudDirectory, cloudFileName, name):
    '''TODO'''
    
    # load cloud data
    cloudData = loadmat(cloudDirectory + "/" + cloudFileName)
    extents = cloudData["extents"].flatten()
    scale = float(cloudData["scale"])
    cloud = cloudData["cloud"]
    cloudTmodel = cloudData["cloudTmodel"]
    
    # load mesh
    body = self.GenerateKinBody(extents, name)
    body.SetName(name)
    body.extents = extents
    body.scale = scale
    body.cloud = cloud
    body.cloudTmodel = cloudTmodel
    self.supportObjects.append(body)
    
    return body
  
  def LoadRandomSupportObject(self, name):
    '''TODO'''
    
    diameter = uniform(self.supportObjectDiameterRange[0], self.supportObjectDiameterRange[1])
    height = uniform(self.supportObjectHeightRange[0], self.supportObjectHeightRange[1])
    return self.GenerateKinBody([diameter, height], name)
    
  def PlaceBottleOnCoasterAtRandom(self):
    '''TODO'''
    
    if len(self.unplacedObjects) == 0 or len(self.supportObjects) == 0:
      return
    
    # find a random bottle-support pair
    bottle = self.unplacedObjects[randint(len(self.unplacedObjects))]
    
    unoccupiedSupports = []
    for support in self.supportObjects:
      if support not in self.occupiedSupports:
        unoccupiedSupports.append(support)
    if len(unoccupiedSupports) == 0: return
    coaster = unoccupiedSupports[randint(len(unoccupiedSupports))]
    
    # determine a random position of the bottle over the coaster
    T = eye(4)
    theta = uniform(0, 2 * pi)
    radius = uniform(0, coaster.extents[0] / 2.0)
    T[0, 3] = coaster.GetTransform()[0, 3] + radius * cos(theta)
    T[1, 3] = coaster.GetTransform()[1, 3] + radius * sin(theta)
    X = point_cloud.Transform(T, bottle.cloud)
    coasterTop = coaster.GetTransform()[2, 3] + coaster.extents[1] / 2.0
    T[2, 3] = coasterTop - min(X[:, 2]) + 0.001
    bottle.SetTransform(dot(T, bottle.cloudTmodel))
    
    # track which bottles have been placed
    self.unplacedObjects.remove(bottle)
    self.placedObjects.append(bottle)    
    self.occupiedSupports.append(coaster)
    
  def PrintString(self, string, wait):
    '''TODO'''
    
    if not wait:
      print(string)
    else:
      raw_input(string)
      
  def RandomizeObjectPose(self, body, workspace):
    '''TODO'''
    
    # choose object orientation
    # (select a downward-facing face from the 6 faces on an axis-aligned bounding box...)
    # (... then choose a random orientation about the gravity axis. Works for bottles.)
    r1 = choice(array([0, 1, 2, 3]) * (pi / 2))
    r2 = choice([pi / 2.0, 0.0], p=[2.0 / 3.0, 1.0 / 3.0])
    r3 = uniform(0, 2 * pi)
    R1 = openravepy.matrixFromAxisAngle([0.0, 0.0, 1.0], r1)
    R2 = openravepy.matrixFromAxisAngle([0.0, 1.0, 0.0], r2)
    R3 = openravepy.matrixFromAxisAngle([0.0, 0.0, 1.0], r3)
    T = dot(R3, dot(R2, R1)) # about fixed frame in order 1, 2, 3
    
    # choose xy position
    T[0:2, 3] = array([ \
      uniform(workspace[0][0], workspace[0][1]),
      uniform(workspace[1][0], workspace[1][1])])
      
    # choose height
    X = point_cloud.Transform(T, body.cloud)
    T[2, 3] = self.GetTableHeight() - min(X[:, 2]) + 0.001
    
    # set transform
    body.SetTransform(dot(T, body.cloudTmodel))
    
    # return transform    
    return T
    
  def RemoveBottle(self, bottle):
    '''TODO'''
    
    if bottle not in self.bottleObjects: return
    self.bottleObjects.remove(bottle)
    self.objects.remove(bottle)
    if bottle in self.placedObjects:
      self.placedObjects.remove(bottle)
    if bottle in self.unplacedObjects:
      self.unplacedObjects.remove(bottle)
    self.env.Remove(bottle)
    
  def RemoveBottleAtRandom(self):
    '''TODO'''
    
    if len(self.bottleObjects) > 0:
      self.RemoveBottle(self.bottleObjects[randint(len(self.bottleObjects))])
  
  def RemoveBottleNearestObject(self):
    '''TODO'''
    
    nBottles = len(self.bottleObjects)
    if nBottles == 0: return
    nCoasters = len(self.supportObjects)
    
    # find centers
    centers = zeros((nBottles + nCoasters, 3))
    for i, obj in enumerate(self.bottleObjects + self.supportObjects):
      baseTmodel = obj.GetTransform()
      modelTcloud = point_cloud.InverseTransform(obj.cloudTmodel)
      baseTcloud = dot(baseTmodel, modelTcloud)
      centers[i, :] = baseTcloud[0:3, 3]
    
    # find the bottle nearest to any other object
    nearestBottle = None; nearestDistance = float('inf')
    for i in xrange(nBottles):
      
      distance = []
      for j in xrange(nBottles + nCoasters):
        if i == j: continue
        distance.append(sum(power(centers[i, :] - centers[j, :], 2)))
      distance = min(distance)
      
      if distance < nearestDistance:
        nearestBottle = self.bottleObjects[i]
        nearestDistance = distance
    
    # remove the nearest bottle
    self.RemoveBottle(nearestBottle)
        
  def RemoveBottleNearestCloud(self, cloud):
    '''TODO'''
    
    nBottles = len(self.bottleObjects)
    if nBottles == 0: return
    
    estimatedCenter = mean(cloud, axis = 0)
    estimatedCenter = tile(estimatedCenter, (nBottles, 1))
    actualCenters = empty((nBottles, 3))
    
    for i, bottle in enumerate(self.bottleObjects):
      baseTmodel = bottle.GetTransform()
      modelTcloud = point_cloud.InverseTransform(bottle.cloudTmodel)
      baseTcloud = dot(baseTmodel, modelTcloud)
      actualCenters[i, :] = baseTcloud[0:3, 3]

    nearestIdx = argmin(sum(power(estimatedCenter - actualCenters, 2), axis = 1))
    self.RemoveBottle(self.bottleObjects[nearestIdx])
    
  def ResetScene(self):
    '''TODO'''
    
    super(EnvironmentBottles, self).ResetScene()
    self.bottleObjects = []
    self.supportObjects = []
    self.occupiedSupports = []