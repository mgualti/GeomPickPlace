'''A simulated environment for the open dimension bin packing problem.'''

# python
import os
import re
import fnmatch
# scipy
from scipy.io import loadmat
from numpy.random import randint, uniform
from numpy import cos, cross, dot, eye, pi, reshape, vstack
# openrave
import openravepy
# self
import point_cloud
from planner import Planner
from environment_pick_place import EnvironmentPickPlace

class EnvironmentPacking(EnvironmentPickPlace):

  def __init__(self, showViewer, showWarnings):
    '''Initialize an EnvironmentPacking instance.
    
    - Input showViewer: If True, shows the OpenRAVE viewer. Set to False if evaluating a large
      number of episodes or if running on a remote machine.
    - Input showWarnings: If True, shows OpenRAVE warnings on initialization.
    '''
    
    EnvironmentPickPlace.__init__(self, showViewer, showWarnings)
    self.planner = Planner(self, 1) # used for finding stable placements
    
    # parameters (grasp simulation)
    self.halfAngleOfGraspFrictionCone = 12.0 * pi / 180.0 # half angle of friction cone, in radians
    self.cosHalfAngleOfGraspFrictionCone = cos(self.halfAngleOfGraspFrictionCone)
    self.graspContactWidth = 0.001 # how far contact points can be from fingers, in meters
    
    # parameters (place simulation)
    self.faceDistTol = 0.004 # how far object can be from table for a stable placement, in meters
    
    # box geometry
    ext = [0.30, 0.30, 0.15, 0.01]
    pos = [-0.40, 0.00, self.GetTableHeight() + ext[2] / 2.0 + ext[3]]
    T = eye(4)
    
    bottom = self.GenerateKinBody([ext[0] + 2 * ext[3], ext[1] + 2 * ext[3], ext[3]], "box-bottom")
    T[0:3, 3] = [pos[0], pos[1], pos[2] - ext[2] / 2.0 ]
    bottom.SetTransform(T)
    
    back = self.GenerateKinBody([ext[3], ext[1], ext[2]], "box-back")
    T[0:3, 3] = [pos[0] - ext[0] / 2.0 - ext[3] / 2.0, pos[1], pos[2]]
    back.SetTransform(T)
    
    front = self.GenerateKinBody([ext[3], ext[1], ext[2]], "box-front")    
    T[0:3, 3] = [pos[0] + ext[0] / 2.0 + ext[3] / 2.0, pos[1], pos[2]]
    front.SetTransform(T)
    
    left = self.GenerateKinBody([ext[0] + 2 * ext[3], ext[3], ext[2]], "box-left")
    T[0:3, 3] = [pos[0], pos[1] - ext[1] / 2.0 - ext[3] / 2.0, pos[2]]
    left.SetTransform(T)
    
    right = self.GenerateKinBody([ext[0] + 2 * ext[3], ext[3], ext[2]], "box-right")
    T[0:3, 3] = [pos[0], pos[1] + ext[1] / 2.0 + ext[3] / 2.0, pos[2]]
    right.SetTransform(T)
    
    self.boxPosition = pos
    self.boxExtents = ext
    self.box = self.objects
    self.objects = []
    self.hasBox = True
    
    for body in self.box:
      body.GetLinks()[0].SetStatic(True)
      body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor([0.75, 0.75, 0.75])
    
  def AddBox(self):
    '''Adds the box back to the environment, if it is missing. (Can remove again with RemoveBox.)'''
    
    if self.hasBox: return
    for part in self.box:
      self.env.Add(part)
    self.hasBox = True

  def EvaluateArrangement(self):
    '''Retrieves information on how well the box is packed.
    
    - Returns nPacked: The number of objects currently packed into the box.
    - Returns packingHeight: The height of tallest object part in the box, in meters, relative to
      the bottom of the box.
    '''
    
    nPacked = len(self.placedObjects)
    packingHeight = -float('inf')
    
    for obj in self.placedObjects:
      cloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      height = max(cloud[:, 2])
      if height > packingHeight:
        packingHeight = height
    
    # set height w.r.t. box bottom rather than w.r.t. world frame
    packingHeight -= (self.GetTableHeight() + self.boxExtents[3])
    
    return nPacked, packingHeight
    
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
    
    isGraspSuccess = []; isGraspAntipodal = []; isTempPlaceStable = []
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
        return False, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
        
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
        isTempPlaceStable.append(self.IsPlacementStable(cloud, self.faceDistTol))
        isStableString = "is" if isTempPlaceStable[-1] else "is not"
        print("Temporary place {} stable.".format(isStableString))
    
    # indicate this object as being placed
    self.placedObjects.append(objInHand)
    self.unplacedObjects.remove(objInHand)
    
    return True, isGraspSuccess, isGraspAntipodal, isTempPlaceStable
    
  def LoadInitialScene(self, nObjects, cloudDirectory, workspace, maxPlaceAttempts = 30):
    '''Randomly selects objects and places them randomly (but stably) in the scene. Attempts to
       ensure no two objects are in contact. Old objects are removed from the OpenRAVE environment
       before new ones are added.
    
    - Input nObjects: Number of objects to add.
    - Input cloudDirectory: Full path to directory containing .mat files with ground truth
      clouds for all objects. (Files generated using generare_full_clouds_packing.py.)
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
    
    # reset the initial scene    
    self.ResetScene()
    self.MoveRobotToHome()

    # load objects
    cloudFileNames = os.listdir(cloudDirectory)
    cloudFileNames = fnmatch.filter(cloudFileNames, "*.mat")
    
    for i in xrange(nObjects):
      
      # choose a random object from the folder
      fileName = cloudFileNames[randint(len(cloudFileNames))]
      
      # load object
      body = self.LoadObjectFromFullCloudFile(cloudDirectory, fileName, "object-{}".format(i))
      
      # compute support faces
      triangles, normals, center = self.planner.GetSupportSurfaces(body.cloud)
    
      # select pose for object
      for j in xrange(maxPlaceAttempts):
        self.PlaceObjectStablyOnTableAtRandom(body, triangles, normals, center, workspace)
        if not self.env.CheckCollision(body): break
    
  def LoadObjectFromFullCloudFile(self, cloudDirectory, cloudFileName, name):
    '''Loads object mesh given full cloud directory and file name.
    
    - Input cloudDirectory: Full path to the directory containing the full cloud .mat files for
      packing (generated with script generate_full_clouds_packing.py).
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
    cloudFileName = cloudDirectory + "/" + cloudFileName
    cloudData = loadmat(cloudFileName)
    scale = float(cloudData["scale"])
    cloud = cloudData["cloud"]
    normals = cloudData["normals"]
    meshFileName = cloudData["meshFileName"][0].encode("ascii")
    
    # load mesh
    self.env.Load(meshFileName, {"scalegeometry":str(scale)})
    body = self.env.GetKinBody(re.findall("/[^/]*.ply$", meshFileName)[0][1:-4])
    if body is None: raise Exception("Failed to load mesh {}".format(meshFileName))
    
    # set color
    colorIdx = len(self.objects) % len(self.colors)
    body.GetLinks()[0].GetGeometries()[0].SetAmbientColor(self.colors[colorIdx])
    
    # save properties
    self.objects.append(body)
    self.unplacedObjects.append(body)
    body.SetName(name)
    body.scale = scale
    body.cloud = cloud
    body.normals = normals
    body.meshFileName = meshFileName
    body.cloudFileName = cloudFileName
    
    return body
    
  def PlaceObjectStablyOnTableAtRandom(self, body, triangles, normals, center, workspace, \
    visualize = False):
    '''Finds a random, stable placement for the object (possibly in collision).
    
    - Input body: KinBody handle to object to place, assumed loaded in the OpenRAVE environment.
    - Input triangles: mx3 integer numpy array, indexing obj.cloud, where each row indicates a facet
      in the convex hull of obj.cloud.
    - Input normals: mx3 numpy array, outward surface normals for each triangle, each row assumed
      to have L2 norm of 1.
    - Input center: mean(obj.cloud, axis = 1).
    - Input workspace: Area to place the object in, with form [(xMin, xMax), (yMin, yMax)]. Height
      is inferred from the table height.
    - Input visualize: If True, shows the triangle in contact with the table.
    - Returns invT: The new pose of the object in the base frame.
    '''
    
    # input checking
    if len(triangles.shape) != 2 or triangles.shape[1] != 3:
      raise Exception("Expected triangles to be 3D; got shape{}.".format(triangles.shape))
    
    if len(normals.shape) != 2 or normals.shape[1] != 3:
      raise Exception("Expected normals to be 3D; got shape {}.".format(normals.shape))
      
    if triangles.shape[0] != normals.shape[0]:
      raise Exception("Expected triangles (length {}) and normals (length {}) to have same " + \
        "number of elements.".format(triangles.shape[0], normals.shape[0]))
      
    if len(center) != 3:
      raise Exception("Expected center to have 3 elements; got {}.".format(len(center)))
    
    if not isinstance(visualize, bool):
      raise Exception("Expected visualize to have type bool; got {}.".format(type(visualize)))
    
    # choose a face and a rotation about that face's normal
    idx = randint(len(triangles))
    faceNormal = -normals[idx]
    theta = uniform(0, 2 * pi)
    
    # define a coordinate system where the z-axis is faceNormal, the position is the cloud's center,
    # and there is a rotation of theta about the z-axis.
    T = eye(4)
    T[0:3, 3] = center
    T[0:3, 2] = faceNormal
    R = openravepy.matrixFromAxisAngle(faceNormal, theta)[0:3, 0:3]
    T[0:3, 1] = dot(R, self.planner.GetOrthogonalUnitVector(T[0:3, 2]))
    T[0:3, 0] = cross(T[0:3, 1], T[0:3, 2])
    
    # check the point cloud in this new coordinate system
    invT = point_cloud.InverseTransform(T)
    cloudRotated = point_cloud.Transform(invT, body.cloud)
      
    # determine translation
    invT[0, 3] += uniform(workspace[0][0], workspace[0][1])
    invT[1, 3] += uniform(workspace[1][0], workspace[1][1])
    invT[2, 3] += self.GetTableHeight() - min(cloudRotated[:, 2]) + 0.001
    
    # set transform
    body.SetTransform(invT)
    
    # visualize place facet
    if visualize:
      self.PlotCloud(point_cloud.Transform(invT, vstack([body.cloud[triangles[idx]], \
        reshape(center, (1, 3))])))
    
    # return transform    
    return invT
      
  def RemoveBox(self):
    '''Removes the box from the environment. (Can add it back with AddBox.)'''
    
    if not self.hasBox: return
    for part in self.box:
      self.env.Remove(part)
    self.hasBox = False