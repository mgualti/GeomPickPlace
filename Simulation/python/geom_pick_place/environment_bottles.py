'''A simulated environment for placing bottles onto coasters.'''

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
from numpy import arange, argmax, argmin, array, copy, cos, cov, dot, empty, eye, hstack, maximum, \
  mean, pi, power, reshape, sin, sum, tile, vstack, zeros
# openrave
import openravepy
# self
import point_cloud
from environment_pick_place import EnvironmentPickPlace

class EnvironmentBottles(EnvironmentPickPlace):

  def __init__(self, showViewer, showWarnings):
    '''Initialize an EnvironmentBottles instance.
    
    - Input showViewer: If True, shows the OpenRAVE viewer. Set to False if evaluating a large
      number of episodes or if running on a remote machine.
    - Input showWarnings: If True, shows OpenRAVE warnings on initialization.
    '''
    
    EnvironmentPickPlace.__init__(self, showViewer, showWarnings)
    
    # parameters (object geometry)
    self.supportObjectHeightRange = [0.005, 0.020] # height of coasters (cylinders), in meters
    self.supportObjectDiameterRange = [0.08, 0.12] # diameter of cylinders (cylinders), in meters
    
    # parameters (grasp simulation)
    self.halfAngleOfGraspFrictionCone = 12.0 * pi / 180.0 # radians
    self.cosHalfAngleOfGraspFrictionCone = cos(self.halfAngleOfGraspFrictionCone)
    self.graspContactWidth = 0.001 # meters
    
    # parameters (place simulation)
    self.placeOrientTolerance = 20.0 * pi / 180.0 # tolerance for bottle "upright", in radians
    self.placeHeightTolerance = [0.02, 0.02] # allowed height [below, above] support, in meters
    self.faceDistTol = 0.004 # how far triangle can be from table for stable placement, in meters
    
    # initialization
    self.bottleObjects = [] # bottles (OpenRAVE handles)
    self.supportObjects = [] # coasters (OpenRAVE handles)
    self.occupiedSupports = [] # coastres with an object placed on them (OpenRAVE handles)
    
  def AddSensorNoise(self, cloud, sigmaRotation, nNeighborhoods, sigmaNeighborhoodRadius,
    sigmaNeighborhoodZ, sigmaPointwiseZ):
    '''Adds simulated noise to point cloud. Includes rotation of the cloud, Z-errors in
       neighborhoods, and point-wise independent Z-errors. Also, the cloud is voxelized to 3 mm.
    
    - Input cloud: nx3 numpy array.
    - Input sigmaRotation: cloud is rotated about an axis, chosen uniformly at random, with angle ~
      N(0, sigmaRotation), where sigmaRotation is a scalar.
    - Input nNeighborhoods: Number of neighborhoods to add noise to.
    - Input sigmaNeighborhoodRadius: Neighborhood radius (in meters) ~ N(0, sigmaNeighborhoodRadius).
      (Negative radii are treated as radius 0. This way, on average, half the neighborhoods are empty.)
    - Input sigmaNeighborhoodZ: Z-offset of neighborhoods (in meters) ~ N(0, sigmaNeighborhoodZ).
    - Input sigmaPointwiseZ: Z-offset of each point independently ~ N(0, sigmaPointwiseZ).
    - Returns cloud: A copy of the input cloud (nx3 numpy array) with added sensor noise.
    '''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("cloud must be 3D. Received shape {}.".format(cloud.shape))
      
    if sigmaRotation < 0:
      raise Exception("sigmaRotation must be non-negative. Received {}.".format(sigmaRotation))
      
    if not isinstance(nNeighborhoods, (int, long)):
      raise Exception("nNeighborhoods must be integer.")
    
    if nNeighborhoods < 0:
      raise Exception("nNeighborhoods must be non-negative. Received {}.".format(nNeighborhoods))
      
    if sigmaNeighborhoodRadius < 0:
      raise Exception("sigmaRotation must be non-negative. Received {}.".format(sigmaRotation))
      
    if sigmaNeighborhoodRadius < 0:
      raise Exception("sigmaNeighborhoodZ must be non-negative. Received {}.".format(\
        sigmaNeighborhoodZ))
      
    if sigmaPointwiseZ < 0:
      raise Exception("sigmaPointwiseZ must be non-negative. Received {}.".format(sigmaPointwiseZ))
      
    if cloud.shape[0] == 0:
      return copy(cloud)
    
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
    
    # voxelize and return result
    cloud = point_cloud.Voxelize(0.003, cloud)
    return cloud
    
  def EvaluateArrangement(self):
    '''Returns the number of correctly placed bottles.'''
    
    return len(self.placedObjects)
    
  def EvaluatePlacement(self, bottle):
    '''Determines if a given bottle is placed correctly.
    
    - Input bottle: OpenRAVE handle to the bottle to check. Assumed in the OpenRAVE environment.
    - Returns placementCorrect: True if the placement is a goal placement and False otherwise.
    - Returns support: If placementCorrect is True, the coaster (OpenRAVE handle) supporting the
      placed bottle. None otherwise.
    '''
    
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
    '''Simulate (and evaluate) a planned sequence of pick-places.
    
    - Input pickPlaces: List of n homogeneous transforms (4x4 numpy arrays), describing hand poses,
      where even-indexed elements are grasps and odd-indexed elements are places.
    - Input plannedConfigs: List of n arm configurations (6-element numpy arrays), assumed to be an
      IK solution for each of the n pickPlaces.
    - Input targObjCloud: Estimated point cloud of the object to be moved, in the world frame.
    - Input showSteps: If True, prompts for user input for each step of the plan, as it is executed.
    - Returns isGraspSuccess: List of boolean values for each attempted grasp, indicating if the
      grasp was successfully executed. Length <= n/2.
    - Returns isGraspAntipodal: List of boolean values for each grasp where the antipodal condition
      was checked, indicating if the grasp was antipodal. Length <= n/2.
    - Returns isTempPlaceStable: List of boolean values for each attempted temporary placement,
      indicating if the temporary placement was successful. (Unlike failed grasps, if temporary
      placements are unstable, execution continues.) Length <= n/2 - 2.
    '''
    
    # input checking
    
    if len(pickPlaces) != len(plannedConfigs):
      raise Exception("pickPlaces (length {}) and plannedConfigs (length {}) must have the " + \
        "same length.".format(len(pickPlaces, len(plannedConfigs))))
        
    if len(targObjCloud.shape) != 2 or targObjCloud.shape[1] != 3:
      raise Exception("targObjCloud must be 3D.")
      
    if not isinstance(showSteps, bool):
      raise Exception("showSteps must be of type bool.")
    
    # some preparation...
    
    prnt = lambda s: self.PrintString(s, showSteps)
    picks = [pickPlaces[i] for i in xrange(0, len(pickPlaces), 2)]
    places = [pickPlaces[i] for i in xrange(1, len(pickPlaces), 2)]
    
    # execute regrasp plan
    
    isGraspSuccess = []; isGraspAntipodal = []; isTempPlaceStable = []
    
    for i, pick in enumerate(picks):
      
      isGraspSuccess.append(False)
      
      # figure out which object is being grasped.
      objsInHand = self.FindObjectsInHand(pick)
      if len(objsInHand) == 0:
        prnt("Nothing in the hand!")
        self.RemoveBottleNearestCloud(targObjCloud)
        return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      if len(objsInHand) > 1:
        prnt("Multiple objects in the hand!")
        self.RemoveBottleNearestCloud(targObjCloud)
        return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      objInHand = objsInHand[0]
      if objInHand not in self.bottleObjects:
        prnt("The object in the hand is not a bottle.")
        self.RemoveBottleNearestCloud(targObjCloud)
        return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      
      # move arm to grasp pose
      self.MoveRobot(plannedConfigs[2 * i + 0])
      prnt("Grasp {} / {}.".format(i + 1, len(picks)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at grasp.")
        self.RemoveBottle(objInHand)
        return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
        
      # check if grasp is antipodal on underlying object
      cloud, normals = point_cloud.Transform(dot(objInHand.GetTransform(), \
        point_cloud.InverseTransform(objInHand.cloudTmodel)), objInHand.cloud, objInHand.normals)
      isAntipodal, _ = self.IsAntipodalGrasp(pick, cloud, normals,
        self.cosHalfAngleOfGraspFrictionCone, self.graspContactWidth)
      isGraspAntipodal.append(isAntipodal)
      if not isAntipodal:
          prnt("Grasp is not antipodal.")
          self.RemoveBottle(objInHand)
          return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      isGraspSuccess[-1] = True
      
      # move arm to place pose
      self.MoveRobot(plannedConfigs[2 * i + 1])
      self.MoveObjectToHandAtGrasp(pick, objInHand)
      prnt("Moved object to place {} / {}".format(i + 1, len(places)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at place.")
        self.RemoveBottle(objInHand)
        return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
        
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
      return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
    
    # (otherwise, the placement is good: add it to the list of placed objects)
    self.placedObjects.append(objInHand)
    self.occupiedSupports.append(support)
    self.unplacedObjects.remove(objInHand)
    return isGraspSuccess, isGraspAntipodal, isTempPlaceStable
    
  def FindObjectsInHand(self, bTh):
    '''Returns a list of objects intersecting the hand's rectangular closing region.
    
    - Input bTh: 4x4 numpy array, assumed to be a homogeneous transform, indicating the pose of the
      hand in the base frame.
    - Returns objectsInHand: Handles (of type KinBody) to objects in the hand.
    '''
    
    # input checking
    if len(bTh.shape) != 2 or bTh.shape[0] != 4 or bTh.shape[1] != 4:
      raise Exception("bTh must be 4x4. Received shape {}.".format(bTh.shape))
    
    # find objects in hand
    objectsInHand = []
    for i, obj in enumerate(self.objects):
      bTo = dot(obj.GetTransform(), point_cloud.InverseTransform(obj.cloudTmodel))
      hTo = dot(point_cloud.InverseTransform(bTh), bTo)
      X = point_cloud.Transform(hTo, obj.cloud)
      X = point_cloud.FilterWorkspace(self.handClosingRegion, X)
      if X.size > 0: objectsInHand.append(obj)
    return objectsInHand
  
  def GetObjectCentroids(self, objects):
    '''Find centroid (average of full cloud) for each object.
    - Input objects: List of objects (KinBody).
    - Returns centers: nObj x 3 centroids.
    '''
    
    centers = empty((len(objects), 3))
    
    for i, obj in enumerate(objects):
      cloud = point_cloud.Transform(dot(obj.GetTransform(),
        point_cloud.InverseTransform(obj.cloudTmodel)), obj.cloud)
      centers[i, :] = mean(cloud, axis = 0)
      
    return centers  
  
  def GetSegmentation(self, cloud, nCategories):
    '''Gets segmentation mask for cloud using ground truth point clouds.
    
    - Input cloud: Point cloud (nx3 numpy array) observation of scene.
    - Input nCategories: Maximum number of objects present in cloud.
    - Returns segmentation: n x nCategories binary matrix indicating which points belong to the same
      object. Each row is 1-hot.
    '''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("cloud must be 3D. Received shape {}.".format(cloud.shape))
      
    if not isinstance(nCategories, (int, long)):
      raise Exception("nCategories must by integer.")
      
    if nCategories < 0:
      raise Exception("nCategories must be non-negative. Received {}.".format(nCategories))
      
    if nCategories < len(self.objects):
      raise Exception("There are fewer categories than objects ({} < {}).".format(\
        nCategories, len(self.objects)))
    
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
    '''Determines if the bottle is upright within some tolerance. Note: This assumes the bottle's
       y-axis is "up", which is true for ShapeNetCore 2.0, but not 3DNet.
    
    - Input bottle: KinBody of bottle to check.
    - Returns isUpright: True if the bottle's up axis is within self.placeOrientTolerance radians
      of the world z axis (negative gravity).
    '''
    
    T = bottle.GetTransform()
    return dot(array([0.0, 0.0, 1.0]), T[0:3, 1]) >= cos(self.placeOrientTolerance)
    
  def IsObjectWidthInLimits(self, cloud, minWidth = 0.030, maxWidth = 0.080):
    '''Project the object onto the x,y plane and and check that the widths are within given limits.
    
    - Input cloud: Point cloud of object (nx3 numpy array), assumed to be oriented upright.
    - Input minWidth: If smallest part is narrower than this (in meters), False is returned.
    - Input maxWidth: If largest part is greater than this (in meters), False is returned.
    - Returns: True if the object width is in the limits and False otherwise.
    '''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("cloud must be 3D. Received shape {}.".format(cloud.shape))
      
    if not isinstance(minWidth, (float, int, long)):
      raise Exception("Expected scalar minWidth.")
      
    if minWidth < 0:
      raise Exception("minWidth must be non-negative.")
      
    if not isinstance(maxWidth, (float, int, long)):
      raise Exception("Expected scalar maxWidth.")
      
    if maxWidth < 0:
      raise Exception("maxWidth must be non-negative.")
    
    # find directions of minimum/maximum variance
    X = cloud[:, 0:2]
    S = cov(X, rowvar = False)
    values, vectors = eig(S)
    widestAxisIdx = argmax(values)
    widestAxis = vectors[:, widestAxisIdx]
    narrowestAxis = vectors[:, widestAxisIdx - 1]
    
    # rotate points so that least variance is in the x direction
    sRb = vstack([reshape(widestAxis, (1, 2)), reshape(narrowestAxis, (1, 2))])
    X = dot(sRb, X.T).T
    #V = hstack([X, zeros((X.shape[0], 1))])
    #point_cloud.Plot(V)
    
    widest = max(X[:, 0]) - min(X[:, 0])
    narrow = max(X[:, 1]) - min(X[:, 1])
    
    return narrow >= minWidth and widest <= maxWidth
    
  def LoadInitialScene(self, nObjects, cloudDirectoryBottles, cloudDirectoryCoasters, workspace,
    maxPlaceAttempts = 30):
    '''Randomly selects objects and places them randomly (but stably) in the scene. Attempts to
       ensure no two objects are in contact. Old objects are removed from the OpenRAVE environment
       and new objects are added in.
    
    - Input nObjects: Number of bottles/coasters to add.
    - Input cloudDirectoryBottles: Full path to directory containing .mat files with ground truth
      clouds for bottles. (Files generated using generare_full_clouds_bottles.py.)
    - Input cloudDirectoryCoasters: Full path to directory containing .mat files with ground truth
      clouds for coasters. (Files generated using generate_full_clouds_bottles.py.)
    - Input workspace: Area to place objects in. Has form [(minX, maxX), (minY, maxY)]. Z value
      is determined by the table height.
    - Input maxPlaceAttempts: Maximum number of times to attempt to place the object collision-free.
    '''
    
    # input checking
    if not isinstance(nObjects, (int, long)):
      raise Exception("nObjects must be integer.")
      
    if nObjects < 0:
      raise Exception("nObjects must be positive.")
      
    if not isinstance(cloudDirectoryBottles, str):
      raise Exception("cloudDirectoryBottles must be a string.")
      
    if not isinstance(cloudDirectoryCoasters, str):
      raise Exception("cloudDirectoryCoasters must be a string.")
      
    if not isinstance(maxPlaceAttempts, (int, long)):
      raise Exception("maxPlaceAttempts must be an integer.")
      
    if maxPlaceAttempts < 0:
      raise Exception("maxPlaceAttempts must be positive.")
    
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
    '''Loads bottle mesh given full cloud directory and file name.
    
    - Input cloudDirectory: Full path to the directory containing the full cloud .mat files for
      bottles (generated with script generate_full_clouds_bottles.py).
    - Input cloudFileName: File name within cloudDirectory pointing to the mesh and cloud to load.
    - Input name: Name to assign to the KinBody object. Must be unique within the OpenRAVE environment.
    - Returns body: KinBody handle to object. Object will be added to the OpenRAVE environment.
    '''
    
    # input checking
    if not isinstance(cloudDirectory, str):
      raise Exception("Expected str for cloudDirectory; got {}.".format(type(cloudDirectory)))
      
    if not isinstance(cloudFileName, str):
      raise Exception("Expected str for cloudFileName; got {}.".format(type(cloudFileName)))
      
    if not isinstance(name, str):
      raise Exception("Expected str for name; got {}.".format(name))
    
    allRaveBodies = self.env.GetBodies()
    for raveBody in allRaveBodies:
      if raveBody.GetName() == name:
        raise Exception("Name {} not unique.".format(name))
    
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
    if body is None: raise Exception("Failed to load mesh {}".format(meshFileName))
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
    '''Loads coaster mesh given full cloud directory and file name.
    
    - Input cloudDirectory: Full path to the directory containing the full cloud .mat files for
      coasters (generated with script generate_full_clouds_bottles.py).
    - Input cloudFileName: File name within cloudDirectory pointing to the mesh and cloud to load.
    - Input name: Name to assign to the KinBody object. Must be unique within the OpenRAVE environment.
    - Returns body: KinBody handle to the support object that was loaded. This body is added to
      the OpenRAVE environment.
    '''
    
    # input checking
    if not isinstance(cloudDirectory, str):
      raise Exception("Expected str for cloudDirectory; got {}.".format(type(cloudDirectory)))
      
    if not isinstance(cloudFileName, str):
      raise Exception("Expected str for cloudFileName; got {}.".format(type(cloudFileName)))
      
    if not isinstance(name, str):
      raise Exception("Expected str for name; got {}.".format(name))
    
    allRaveBodies = self.env.GetBodies()
    for raveBody in allRaveBodies:
      if raveBody.GetName() == name:
        raise Exception("Name {} not unique.".format(name))    
    
    # load cloud data
    cloudData = loadmat(cloudDirectory + "/" + cloudFileName)
    extents = cloudData["extents"].flatten()
    scale = float(cloudData["scale"])
    cloud = cloudData["cloud"]
    cloudTmodel = cloudData["cloudTmodel"]
    
    # load mesh
    body = self.GenerateKinBody(extents, name) # adds body to self.objects
    body.SetName(name)
    body.extents = extents
    body.scale = scale
    body.cloud = cloud
    body.cloudTmodel = cloudTmodel
    self.supportObjects.append(body)
    
    return body
  
  def LoadRandomSupportObject(self, name):
    '''Loads a randomly sized coaster (cylinder) into the environment.
    
    - Input name: A unique name for the support object.
    - Returns body: KinBody handle of the object loaded.
    '''
    
    if not isinstance(name, str):
      raise Exception("Expected str for name; got {}.".format(type(name)))
    
    diameter = uniform(self.supportObjectDiameterRange[0], self.supportObjectDiameterRange[1])
    height = uniform(self.supportObjectHeightRange[0], self.supportObjectHeightRange[1])
    return self.GenerateKinBody([diameter, height], name) # adds body to self.objects
    
  def PlaceBottleOnCoasterAtRandom(self):
    '''Correctly places an unplaced bottle. The unplaced bottle and position over the coaster are
       selected uniformly at random.'''
    
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
      
  def RandomizeObjectPose(self, body, workspace):
    '''Randomize position and orientation of bottle, so it is resting stably, but possibly in
       collision. Selects upright placements with probability 1/3 and side placements with
       probabilty 2/3. Upside-down placements are never selected.
    
    - Input body: OpenRAVE KinBody object (for a bottle).
    - Input workspace: Area to place the object in, with form [(xMin, xMax), (yMin, yMax)]. Height
      is inferred from the table height.
    - Returns T: Homogeneous transform of the object's ground truth cloud to the current, selected
      pose of the object.
    '''
    
    # choose object orientation
    # (Select a downward-facing face from 5 faces on the axis-aligned bounding box [upside-down
    #  excluded], then choose a random orientation about the gravity axis. Works for bottles.)
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
    '''Removes the given KinBody from the environment. (If it's not a bottle, no change is made.)'''
    
    if bottle not in self.bottleObjects: return
    self.bottleObjects.remove(bottle)
    self.objects.remove(bottle)
    if bottle in self.placedObjects:
      self.placedObjects.remove(bottle)
    if bottle in self.unplacedObjects:
      self.unplacedObjects.remove(bottle)
    self.env.Remove(bottle)
    
  def RemoveBottleAtRandom(self):
    '''Randomly select a bottle and remove it from the environment.'''
    
    if len(self.bottleObjects) > 0:
      self.RemoveBottle(self.bottleObjects[randint(len(self.bottleObjects))])
  
  def RemoveBottleNearestObject(self):
    '''Remove a bottle whose origin is nearest another object's origin.'''
    
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
    '''Remove the bottle nearest the center of the given point cloud.
    
    - Input cloud: Cloud (nx3 numpy array with n >= 1) in world frame, the nearest bottle to which
      will be removed.
    '''
    
    if len(cloud.shape) != 2 and cloud.shape[1] != 3:
      raise Exception("Expected 3D cloud; got shape {}.".format(cloud.shape))
      
    if cloud.shape[0] == 0:
      raise Exception("Unable to compare distance to empty cloud.")
    
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
    '''Clear all bottles and coasters from the environment.'''
    
    super(EnvironmentBottles, self).ResetScene()
    self.bottleObjects = []
    self.supportObjects = []
    self.occupiedSupports = []