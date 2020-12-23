'''TODO'''

# python
import os
import re
import fnmatch
# scipy
from scipy.io import loadmat
from scipy.spatial import cKDTree
from numpy.random import randint, uniform
from numpy import argmin, arange, cos, cross, dot, eye, hstack, pi, reshape, vstack, zeros
# openrave
import openravepy
# self
import point_cloud
from planner import Planner
from environment_pick_place import EnvironmentPickPlace

class EnvironmentPacking(EnvironmentPickPlace):

  def __init__(self, showViewer, showWarnings):
    '''TODO'''
    
    EnvironmentPickPlace.__init__(self, showViewer, showWarnings)
    self.planner = Planner(self, 1, 0)
    
    # parameters (grasp simulation)
    self.halfAngleOfGraspFrictionCone = 12.0 * pi / 180.0 # radians
    self.cosHalfAngleOfGraspFrictionCone = cos(self.halfAngleOfGraspFrictionCone)
    self.graspContactWidth = 0.001 # meters
    
    # parameters (place simulation)
    self.faceDistTol = 0.004
    
    # box geometry
    ext = [0.27, 0.32, 0.17, 0.01]
    pos = [0.756, -0.027, self.GetTableHeight() + ext[2] / 2.0 + ext[3]]
    T = eye(4)
    
    bottom = self.GenerateKinBody([ext[0] + 2 * ext[3], ext[1] + 2 * ext[3], ext[3]], "box-bottom")
    T[0:3, 3] = [pos[0], pos[1], pos[2] - ext[2] / 2.0 ]
    bottom.SetTransform(T)
    
    back = self.GenerateKinBody([ext[3], ext[1], ext[2]], "box-back")
    T[0:3, 3] = [pos[0] - ext[0] / 2.0 - ext[3] / 2.0, pos[1], pos[2]]
    back.SetTransform(T)
    
    front = self.GenerateKinBody([ext[3], ext[1], ext[2]], "box-back")    
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
    '''TODO'''
    
    if self.hasBox: return
    for part in self.box:
      self.env.Add(part)
    self.hasBox = True
      
  def CheckHandObjectCollision(self, grasps, cloud, cubeSize = 0.005):
    '''TODO'''
    
    # check for empty input
    if len(grasps) == 0:
      return zeros(0, dtype='bool')
      
    # temporarily remove objects from environment (will put them back later)
    hadRobot = self.hasRobot
    hadTable = self.hasTable
    hadBox = self.hasBox
    hadFloatingHand = self.hasFloatingHand
    self.RemoveRobot()
    self.RemoveTable()
    self.RemoveBox()
    self.RemoveObjectSet(self.objects)
    
    # add in the hand model and point cloud
    self.AddFloatingHand()
    self.AddObstacleCloud(cloud, cubeSize)
    
    # check collision
    collision = zeros(len(grasps), dtype='bool')
    for i, grasp in enumerate(grasps):
      self.MoveFloatingHandToPose(grasp)
      collision[i] = self.env.CheckCollision(self.floatingHand)
      
    # remove obstacle cloud
    self.RemoveObstacleCloud()
    
    # add objects back
    if not hadFloatingHand: self.RemoveFloatingHand()
    if hadBox: self.AddBox()
    if hadTable: self.AddTable()
    if hadRobot: self.AddRobot()
    for obj in self.objects:
      self.env.AddKinBody(obj)
    
    # return result
    return collision
    
  def EvaluateArrangement(self):
    '''TODO'''
    
    nPacked = len(self.placedObjects)
    packingHeight = -float('inf')
    
    for obj in self.placedObjects:
      cloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      height = max(cloud[:, 2])
      if height > packingHeight:
        packingHeight = height
    
    # remove table and box height so that 0 is the best possible packing height
    packingHeight -= (self.GetTableHeight() + self.boxExtents[3])
    
    return nPacked, packingHeight
    
  def ExecuteRegraspPlan(self, pickPlaces, plannedConfigs, targObjCloud, showSteps):
    '''TODO'''
    
    prnt = lambda s: self.PrintString(s, showSteps)
    picks = [pickPlaces[i] for i in xrange(0, len(pickPlaces), 2)]
    places = [pickPlaces[i] for i in xrange(1, len(pickPlaces), 2)]
    
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
      self.MoveRobot(plannedConfigs[2 * i])
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
    
  def GetSegmentation(self, cloud, nCategories):
    '''TODO'''
    
    # construct a KD-tree with the full cloud of each object
    fullCloudTrees = []
    for obj in self.unplacedObjects:
      objCloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      fullCloudTrees.append(cKDTree(objCloud))
    
    # find the minimum distance between cloud points and object cloud points
    distances = []
    for i, obj in enumerate(self.unplacedObjects):
      d, _ = fullCloudTrees[i].query(cloud)
      distances.append(reshape(d, (cloud.shape[0], 1)))
    distances = hstack(distances)
    
    # classify points based on the nearest object point
    segmentation = zeros((cloud.shape[0], nCategories), dtype = 'bool')
    rowIdx = arange(cloud.shape[0])
    colIdx = argmin(distances, axis = 1)    
    segmentation[rowIdx, colIdx] = 1
    
    return segmentation
    
  def LoadInitialScene(self, nObjects, cloudDirectory, workspace, maxPlaceAttempts = 30):
    '''TODO'''
    
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
    '''TODO'''
    
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
    '''TODO'''
    
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
    '''TODO'''
    
    if not self.hasBox: return
    for part in self.box:
      self.env.Remove(part)
    self.hasBox = False