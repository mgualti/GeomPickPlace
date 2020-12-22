'''A simulated environment for arranging blocks.'''

# python
import os
import fnmatch
from copy import copy
# scipy
from scipy.io import loadmat
from numpy.random import randint, uniform
from numpy import argmax, array, cos, dot, pi
# openrave
import openravepy
# self
import point_cloud
from environment_pick_place import EnvironmentPickPlace

class EnvironmentBlocks(EnvironmentPickPlace):

  def __init__(self, showViewer, showWarnings):
    '''Initialize an EnvironmentBlocks instance.
    
    - Input showViewer: If True, shows the OpenRAVE viewer. Set to False if evaluating a large
      number of episodes or if running on a remote machine.
    - Input showWarnings: If True, shows OpenRAVE warnings on initialization.
    '''
    
    EnvironmentPickPlace.__init__(self, showViewer, showWarnings)
    
    # parameters (grasp simulation)
    self.halfAngleOfGraspFrictionCone = 5.0 * pi / 180.0 # half angle of friction cone, in radians
    self.cosHalfAngleOfGraspFrictionCone = cos(self.halfAngleOfGraspFrictionCone)
    self.graspContactWidth = 0.001 # how far contact points can be from fingers, in meters
    
    # parameters (place simulation)
    self.faceDistTol = 0.004 # how far object can be from table for a stable placement, in meters
    
  def GenerateRandomBlock(self, extents, name):
    '''Generates a block with dimensions selected uniformly at random.
    
    - Input extents: Bounds on the size of the block, of form
      [(xMin, xMax), (yMin, yMax), (zMin, zMax)].
    - Input name: Name to assign the block. Must be different from the names of all other objects
      currently in the environment.
    '''

    extents = [uniform(extents[0][0], extents[0][1]),
               uniform(extents[1][0], extents[1][1]),
               uniform(extents[2][0], extents[2][1])]
   
    return self.GenerateKinBody(extents, name)
    
  def EvaluateArrangement(self):
    '''Determines how well the currently placed blocks are arranged from tallest to shortest.
    
    - Returns nPlaced: The number of blocks that were placed.
    - Returns orderCorrect: For each placed block, indicates if the length of the longest side
      of the previously placed block is greater than the length of the longest side of this block.
      For the first block, this is always 1. Binary array of length nPlaced.
    - Returns longestEndUp: For each placed block, indicates whether the longest side is aligned
      with the z-axis (i.e., with gravity). Binary array of length nPlaced.
    '''
    
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
      orderCorrect.append(lastHeight >= lengths[longestSide])
      longestEndUp.append(longestSide == up)
      lastHeight = lengths[longestSide]
    
    return nPlaced, orderCorrect, longestEndUp
    
  def ExecuteRegraspPlan(self, pickPlaces, plannedConfigs, targObjCloud, showSteps):
    '''Simulate (and evaluate) a planned sequence of pick-places.
    
    - Input pickPlaces: List of n homogeneous transforms (4x4 numpy arrays), describing hand poses,
      where even-indexed elements are grasps and odd-indexed elements are places.
    - Input plannedConfigs: List of n arm configurations (6-element numpy arrays), assumed to be an
      IK solution for each of the n pickPlaces.
    - Input targObjCloud: Estimated point cloud of the object to be moved, in the world frame.
    - Input showSteps: If True, prompts for user input for each step of the plan, as it is executed.
    - Returns success: True if the plan was executed successfully and the object was placed.
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
      
    # some preparation
    
    prnt = lambda s: self.PrintString(s, showSteps)
    picks = [pickPlaces[i] for i in xrange(0, len(pickPlaces), 2)]
    places = [pickPlaces[i] for i in xrange(1, len(pickPlaces), 2)]
    
    # execute regrasp plan
    
    isTempPlaceStable = []; isGraspAntipodal = []; isGraspSuccess = []
    
    for i, pick in enumerate(picks):
      
      isGraspSuccess.append(False)
      
      # figure out which object is being grasped
      objsInHand = self.FindObjectsInHand(pick)
      if len(objsInHand) == 0:
        prnt("Nothing in the hand!")
        self.RemoveObjectNearestCloud(targObjCloud)
        return False, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      if len(objsInHand) > 1:
        prnt("Multiple objects in the hand!")
        self.RemoveObjectNearestCloud(targObjCloud)
        return False, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      objInHand = objsInHand[0]
      
      # move arm to grasp pose
      self.MoveRobot(plannedConfigs[2 * i + 0])
      prnt("Grasp {} / {}.".format(i + 1, len(picks)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at grasp.")
        self.RemoveUnplacedObject(objInHand)
        return False, isGraspSuccess, isGraspAntipodal, isTempPlaceStable, 
        
      # check if grasp is antipodal on underlying object
      cloud, normals = point_cloud.Transform( \
        objInHand.GetTransform(), objInHand.cloud, objInHand.normals)
      isAntipodal, _ = self.IsAntipodalGrasp(pick, cloud, normals,
        self.cosHalfAngleOfGraspFrictionCone, self.graspContactWidth)
      isGraspAntipodal.append(isAntipodal)
      if not isAntipodal:
          prnt("Grasp is not antipodal.")
          self.RemoveUnplacedObject(objInHand)
          return False, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
      isGraspSuccess[-1] = True
        
      # move arm to place pose
      self.MoveRobot(plannedConfigs[2 * i + 1])
      self.MoveObjectToHandAtGrasp(pick, objInHand)
      prnt("Moved object to place {} / {}".format(i + 1, len(places)))
      if self.env.CheckCollision(self.robot):
        prnt("Arm is in collision at place.")
        self.RemoveUnplacedObject(objInHand)
        return False, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
        
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
    
    return True, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
    
  def LoadInitialScene(self, nObjects, cloudDirectory, workspace, maxPlaceAttempts = 30):
    '''Randomly selects blocks and places them randomly (but stably) in the scene. Attempts to
       ensure no two blocks are in contact. Old blocks are removed from the OpenRAVE environment
       before new blocks are added.
    
    - Input nObjects: Number of blocks to add.
    - Input cloudDirectory: Full path to directory containing .mat files with ground truth
      clouds for blocks. (Files generated using generare_full_clouds_blocks.py.)
    - Input workspace: Area to place objects in. Has form [(minX, maxX), (minY, maxY)]. Z value
      is determined by the table height.
    - Input maxPlaceAttempts: Maximum number of times to attempt to place the object collision-free.
    '''
    
    # input checking
    if not isinstance(nObjects, (int, long)):
      raise Exception("nObjects must be integer.")
      
    if nObjects < 0:
      raise Exception("nObjects must be positive.")
      
    if not isinstance(cloudDirectory, str):
      raise Exception("cloudDirectory must be a string.")
      
    if not isinstance(maxPlaceAttempts, (int, long)):
      raise Exception("maxPlaceAttempts must be an integer.")
      
    if maxPlaceAttempts < 0:
      raise Exception("maxPlaceAttempts must be positive.")
    
    # reset initial scene
    self.ResetScene()
    self.MoveRobotToHome()
    
    # get file names in cloud directory
    cloudFileNames = os.listdir(cloudDirectory)
    cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
    
    for i in xrange(nObjects):
      
      # choose a random object from the folder
      fileName = cloudFileNames[randint(len(cloudFileNames))]
      
      # load object
      body = self.LoadObjectFromFullCloudFile(cloudDirectory, fileName, "object-{}".format(i))
      
      # randomize object's pose, accepting it if it is collision-free
      for j in xrange(maxPlaceAttempts):
        self.RandomizeObjectPose(body, workspace)
        if not self.env.CheckCollision(body): break
          
  def LoadObjectFromFullCloudFile(self, cloudDirectory, cloudFileName, name):
    '''Generates block mesh given full cloud directory and file name.
    
    - Input cloudDirectory: Full path to the directory containing the full cloud .mat files for
      blocks (generated with script generate_full_clouds_blocks.py).
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
    
    # load cloud data
    data = loadmat(cloudDirectory + "/" + cloudFileName)
    
    # generate mesh
    body = self.GenerateKinBody(data["extents"].flatten(), name)
    body.cloud = data["cloud"]
    body.normals = data["normals"]
    self.unplacedObjects.append(body)
    
    return body
    
  def RandomizeObjectPose(self, body, workspace):
    '''Place the block in a randomly generated pose which is stable w.r.t. the table but possibly
       in collision with other objects.
    
    - Input body: OpenRAVE KinBody object (for a bottle).
    - Input workspace: Area to place the object in, with form [(xMin, xMax), (yMin, yMax)]. Height
      is inferred from the table height.
    - Returns T: Homogeneous transform of the object's ground truth cloud to the current, selected
      pose of the object.
    '''
    
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
    T[2, 3] = self.GetTableHeight() + copy(body.extents[upIdx]) / 2.0 + 0.001
    
    # set transform
    body.SetTransform(T)
    
    # return transform    
    return T