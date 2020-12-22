'''TODO'''

# python
import re
# scipy
from numpy.linalg import norm
from numpy.random import randint
from scipy.spatial import cKDTree, ConvexHull
from numpy import abs, arange, argmax, argmin, array, ascontiguousarray, cross, dot, empty, eye, \
  hstack, logical_and, logical_not, max, mean, nonzero, power, repeat, reshape, sort, sum, tile, vstack, zeros
# openrave
import openravepy
# self
import point_cloud
import c_extensions
from environment import Environment
from hand_descriptor import HandDescriptor

class EnvironmentPickPlace(Environment):

  def __init__(self, showViewer, showWarnings):
    '''TODO'''
    
    Environment.__init__(self, showViewer, showWarnings)
    
    # parameters
    self.colors = array([ \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ])
      
    self.viewPointsForNormals = array(
      [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype = "float")
    
    # initialization
    self.InitializeHandRegions()
    self.objects = []
    self.unplacedObjects = []
    self.placedObjects = []
    
  def AreAnyReachable(self, grasps, ignoreObjects = False):
    '''TODO'''
    
    # check for empty input
    if len(grasps) == 0:
      return False
    
    if ignoreObjects:
      # temporarily remove objects from environment (will put them back later)
      self.RemoveObjectSet(self.objects)
    
    # quick collision filtering
    reachable = empty(len(grasps), dtype='bool')
    for i, T in enumerate(grasps):
      reachable[i] = not self.IsHandUnderTable(T)
    
    # check for IK solutions
    configs = [[]] * len(grasps)
    for i, T in enumerate(grasps):
      if not reachable[i]: continue
      configs[i] = self.CalcIkForT(T)
      reachable[i] = len(configs[i]) > 0
    
    # check collisions with the robot
    for i, T in enumerate(grasps):
      if not reachable[i]: continue
      reachable[i] = False
      for config in configs[i]:
        self.MoveRobot(config)
        if not self.env.CheckCollision(self.robot):
          reachable[i] = True
          break
      if reachable[i]:
        break
    
    if ignoreObjects:
      # add objects back
      for obj in self.objects:
        self.env.AddKinBody(obj)
      
    return reachable.any()
    
  def CheckHandObjectCollision(self, grasps, cloud, cubeSize = 0.005):
    '''TODO'''
    
    # check for empty input
    if len(grasps) == 0:
      return empty(0, dtype='bool')
      
    # temporarily remove objects from environment (will put them back later)
    hadRobot = self.hasRobot
    hadTable = self.hasTable
    hadFloatingHand = self.hasFloatingHand
    self.RemoveRobot()
    self.RemoveTable()
    self.RemoveObjectSet(self.objects)
    
    # add in the hand model and point cloud
    self.AddFloatingHand()
    self.AddObstacleCloud(cloud, cubeSize)
    
    # check collision
    collision = empty(len(grasps), dtype='bool')
    for i, grasp in enumerate(grasps):
      self.MoveFloatingHandToPose(grasp)
      collision[i] = self.env.CheckCollision(self.floatingHand)
      
    # remove obstacle cloud
    self.RemoveObstacleCloud()
    
    # add objects back
    if not hadFloatingHand: self.RemoveFloatingHand()
    if hadTable: self.AddTable()
    if hadRobot: self.AddRobot()
    for obj in self.objects:
      self.env.AddKinBody(obj)
    
    # return result
    return collision
      
  def CheckReachability(self, grasps, ignoreObjects = False, obstacleCloud = None, cubeSize = 0.01):
    '''TODO'''
    
    # check for empty input
    if len(grasps) == 0:
      return empty(0, dtype='bool')
    
    if obstacleCloud is not None and obstacleCloud.shape[0] == 0:
      obstacleCloud = None
    
    if ignoreObjects:
      # temporarily remove objects from environment (will put them back later)
      self.RemoveObjectSet(self.objects)
    
    # add the point cloud as an obstacle
    if obstacleCloud is not None:
      self.AddObstacleCloud(obstacleCloud, cubeSize)
    
    # quick collision filtering
    reachable = empty(len(grasps), dtype='bool')
    for i, T in enumerate(grasps):
      reachable[i] = not self.IsHandUnderTable(T)
    
    # check for IK solutions
    configs = [[]] * len(grasps)
    for i, T in enumerate(grasps):
      if not reachable[i]: continue
      configs[i] = self.CalcIkForT(T)
      reachable[i] = len(configs[i]) > 0
      
    # filter grasps which contain obstacles
    if obstacleCloud is not None:
      for i, T in enumerate(grasps):
        if not reachable[i]: continue
        X = point_cloud.Transform(point_cloud.InverseTransform(T), obstacleCloud)
        X = point_cloud.FilterWorkspace(self.handClosingRegion, X)
        if X.size > 0: reachable[i] = False
    
    # check collisions with the robot
    for i, T in enumerate(grasps):
      if not reachable[i]: continue
      reachable[i] = False
      for config in configs[i]:
        self.MoveRobot(config)
        if not self.env.CheckCollision(self.robot):
          reachable[i] = True
          break
    
    # remove the obstacle cloud
    self.RemoveObstacleCloud()
    
    if ignoreObjects:
      # add objects back
      for obj in self.objects:
        self.env.AddKinBody(obj)
      
    return reachable
    
  def EstimateNormals(self, clouds):
    '''Estimate object surface normals for completed clouds, where viewpoints are unknown.
    - Input clouds: List of m, nx3 numpy arrays.
    - Returns normals: List of m, nx3 numpy arrays with rows normalized.
    '''
    
    # input checking
    
    if not isinstance(clouds, list):
      raise Exception("Expected list for clouds. Got {} instead".format(type(clouds)))
    
    for i, cloud in enumerate(clouds):
      if cloud.shape[1] != 3:
        raise Exception("Cloud {} not 3D.".format(i))
    
    # estimate normals
    
    normals = []
    for cloud in clouds:
      
      # Estimate the smallest spacing between points that blocks all rays through the object.
      cloudTree = cKDTree(cloud)
      d, _ = cloudTree.query(cloud, k = 2)
      d = sort(max(d, axis = 1))
      maxDistToRay = d[int(0.9 * d.size)]
      
      # Center view points on the cloud.
      center = mean(cloud, axis = 0)
      center = repeat(array([center]), self.viewPointsForNormals.shape[0], axis = 0)
      centeredViewPoints = self.viewPointsForNormals + center
      
      # Find occlusion for each view point
      cloud = ascontiguousarray(cloud, dtype = "float32")
      centeredViewPoints = ascontiguousarray(centeredViewPoints, dtype = "float32")
      occluded = zeros((cloud.shape[0], centeredViewPoints.shape[0]), dtype = "bool", order = "C")
      c_extensions.IsOccluded(cloud, centeredViewPoints, cloud.shape[0],
        centeredViewPoints.shape[0], maxDistToRay, occluded)
      visible = logical_not(occluded)
      
      # visualization
      #for i, view in enumerate(centeredViewPoints):
      #  print("{} points visible from {}".format(sum(visible[:, i]), view))
      #  point_cloud.Plot(vstack([cloud[visible[:, i], :], view]), )
      
      # Associate a view point to each point in the cloud based on visiblilty from that view point.
      viewPoints = zeros(cloud.shape)
      for i, view in enumerate(centeredViewPoints):
        viewPoints[visible[:, i], :] = tile(reshape(centeredViewPoints[i, :], (1, 3)),
          (sum(visible[:, i]), 1))
        
      # For undetermind view points, set using nearest neighbor that has a view point.
      undetermindViews = sum(visible, axis = 1) == 0
      determindViews = logical_not(undetermindViews)
      determindTree = cKDTree(cloud[determindViews, :])
      _, idx = determindTree.query(cloud[undetermindViews, :])
      viewPoints[undetermindViews, :] = viewPoints[idx, :]
      
      # Estimate surface normals
      normals.append(point_cloud.ComputeNormals(\
        cloud, viewPoints, kNeighbors = 30, rNeighbors = -1))
      
      # visualization
      #point_cloud.Plot(cloud, normals[-1], 1)
      
    return normals
    
  def FindObjectsInHand(self, bTh):
    '''Returns a list of objects intersecting the hand's rectangular closing region.
    - Input bTh: 4x4 homogeneous transform indicating the pose of the hand in the base frame.
    - Returns objectsInHand: Handles (of type KinBody) to objects in the hand.
    '''
    
    objectsInHand = []
    for i, obj in enumerate(self.unplacedObjects):
      bTo = obj.GetTransform()
      hTo = dot(point_cloud.InverseTransform(bTh), bTo)
      X = point_cloud.Transform(hTo, obj.cloud)
      X = point_cloud.FilterWorkspace(self.handClosingRegion, X)
      if X.size > 0: objectsInHand.append(obj)
    return objectsInHand
    
  def GetReachableConfigs(self, grasps, ignoreObjects = False, obstacleCloud = None, cubeSize = 0.01):
    '''TODO'''
    
    # check for empty input
    if len(grasps) == 0:
      return [], empty(0)
      
    if obstacleCloud is not None and obstacleCloud.shape[0] == 0:
      obstacleCloud = None
    
    if ignoreObjects:
      # temporarily remove objects from environment (will put them back later)
      self.RemoveObjectSet(self.objects)
    
    # add the point cloud as an obstacle
    if obstacleCloud is not None:
      self.AddObstacleCloud(obstacleCloud, cubeSize)
    
    # quick collision filtering
    reachable = empty(len(grasps), dtype='bool')
    for i, T in enumerate(grasps):
      reachable[i] = not self.IsHandUnderTable(T)
    
    # check for IK solutions
    configs = [[]] * len(grasps)
    for i, T in enumerate(grasps):
      if not reachable[i]: continue
      configs[i] = self.CalcIkForT(T)
      reachable[i] = len(configs[i]) > 0
      
    # filter grasps which contain obstacles
    if obstacleCloud is not None:
      for i, T in enumerate(grasps):
        if not reachable[i]: continue
        X = point_cloud.Transform(point_cloud.InverseTransform(T), obstacleCloud)
        X = point_cloud.FilterWorkspace(self.handClosingRegion, X)
        if X.size > 0: reachable[i] = False
    
    # check collisions with the robot
    for i, T in enumerate(grasps):
      if not reachable[i]: continue
      collisionFreeConfigs = []
      for config in configs[i]:
        self.MoveRobot(config)
        if not self.env.CheckCollision(self.robot):
          collisionFreeConfigs.append(config)
      configs[i] = collisionFreeConfigs
      reachable[i] = len(configs[i]) > 0
    
    # remove the obstacle cloud
    if obstacleCloud is not None:
      self.RemoveObstacleCloud()
    
    if ignoreObjects:
      # add objects back
      for obj in self.objects:
        self.env.AddKinBody(obj)
      
    return configs, reachable
    
  def GetObjectCentroids(self, objects):
    '''Find centroid (average of full cloud) for each object.
    - Input objects: List of objects (KinBody).
    - Returns centers: nObj x 3 centroids.
    '''
    
    centers = empty((len(objects), 3))
    
    for i, obj in enumerate(objects):
      cloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      centers[i, :] = mean(cloud, axis = 0)
      
    return centers
    
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
    
  def InitializeHandRegions(self):
    '''Determines hand geometry, in the descriptor reference frame, for collision checking. Should
       be called once at initialization.
    '''
    
    # find default descriptor geometry
    desc = HandDescriptor(eye(4))
    
    # important reference points
    topUp = desc.top + (desc.height / 2) * desc.axis
    topDn = desc.top - (desc.height / 2) * desc.axis
    BtmUp = desc.top + (desc.height / 2) * desc.axis
    BtmDn = desc.top - (desc.height / 2) * desc.axis
    
    # cuboids representing hand regions, in workspace format
    self.handClosingRegion = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width  / 2, desc.width  / 2),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handFingerRegionL = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width / 2 - 0.01, -desc.width  / 2),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handFingerRegionR = [
      (-desc.height / 2, desc.height / 2),
      (desc.width / 2, desc.width / 2 + 0.01),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handTopRegion = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width / 2 - 0.01, desc.width / 2 + 0.01),
      (desc.depth  / 2, desc.depth  / 2 + 0.01)]
      
    # find corners of hand collision geometry
    self.externalHandPoints = array([ \
      topUp + ((desc.width / 2) + 0.01) * desc.binormal,
      topUp - ((desc.width / 2) + 0.01) * desc.binormal,
      topDn + ((desc.width / 2) + 0.01) * desc.binormal,
      topDn - ((desc.width / 2) + 0.01) * desc.binormal,
      BtmUp + ((desc.width / 2) + 0.01) * desc.binormal,
      BtmUp - ((desc.width / 2) + 0.01) * desc.binormal,
      BtmDn + ((desc.width / 2) + 0.01) * desc.binormal,
      BtmDn - ((desc.width / 2) + 0.01) * desc.binormal, ])
      
  def FilterWorkspaceWithIndices(self, workspace, cloud, normals):
    '''TODO'''

    mask = (((((cloud[:,0] >= workspace[0][0])  & (cloud[:,0] <= workspace[0][1])) \
             & (cloud[:,1] >= workspace[1][0])) & (cloud[:,1] <= workspace[1][1])) \
             & (cloud[:,2] >= workspace[2][0])) & (cloud[:,2] <= workspace[2][1])
    
    cloud = cloud[mask, :]
    normals = normals[mask, :]
    return cloud, normals, nonzero(mask)[0]
    
  def IsAntipodalGrasp(self, bTh, cloud, normals, cosHalfAngleOfFrictionCone, contactWidth):
    '''Checks if a grasp is antipodal. See spatial antipodal grasps in "A Mathematical Introduction
       to Robotic Manipulation" by Murray, Li, and Sastry.
       
    - Input bTh: Gripper pose (4x4 homogeneous transform) in the world frame.
    - Input cloud: nx3 numpy array of points representing the object in the world frame.
    - Input normals: nx3 numpy array of surface normals for each corresponding point in cloud;
      assumed normalized across rows.
    - Input cosHalfAngleOfFrictionCone = cos(theta / 2), where theta is the solid angle (in radians)
      defining the size of the friction cone formed between the hand and the object when fingers
      are in contact. 
    - Input contactWidth: If, after the fingers close up to the first points encountered, a point is
      less than distance contactWidth from the fingers, this point is also considered in contact.
      This is to account for small errors in numerical precision, noise in the point cloud, or
      object softness. Set to 0 to adhere to the strict definition of spatial antipodal grasps.
    - Returns isAntipodal: True if the grasp is considered to be spatial antipodal, according to
      the above-described relaxation, or False otherwise.
    - Returns contacts: A pair of indices into cloud indicating which points are in contact.
    '''
    
    # input checking
    if len(bTh.shape) != 2 or bTh.shape[0] != 4 or bTh.shape[1] != 4:
      raise Exception("bTh must be a homogeneous transform (received shape {}).".format(bTh.shape))
      
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("Cloud must be 3D.")
      
    if len(normals.shape) != 2 or normals.shape[1] != 3:
      raise Exception("Normals must be 3D.")
      
    if cloud.shape[0] != normals.shape[0]:
      raise Exception("cloud has {} points while normals has {} points.".format(
        cloud.shape[0], normals.shape[0]))
    
    if not isinstance(cosHalfAngleOfFrictionCone, float):
      raise Exception("cosHalfAngleOfFrictionCone must be of type float (received {}).".format(
        type(cosHalfAngleOfFrictionCone)))
    
    if cosHalfAngleOfFrictionCone < 0 or cosHalfAngleOfFrictionCone > 1.0:
      raise Exception("cosFrictionCone must be in [0, 1] (received {}).".format(
        cosHalfAngleOfFrictionCone))
    
    if not isinstance(contactWidth, float):
      raise Exception("contactWidth must be of type float (received {}).".format(
        type(contactWidth)))
    
    if contactWidth < 0:
      raise Exception("contactWidth must be non-negative (received {}).".format(contactWidth))

    # put cloud into hand reference frame
    hTb = point_cloud.InverseTransform(bTh)
    X, N = point_cloud.Transform(hTb, cloud, normals)
    X, N, inHandIdxs = self.FilterWorkspaceWithIndices(self.handClosingRegion, X, N)
    if X.shape[0] == 0: return False, zeros(0, dtype = "int") # no points in the hand

    # find the actual contact points
    leftPointIdx = argmin(X[:, 1])
    rightPointIdx = argmax(X[:, 1])
    leftPoint = X[leftPointIdx, 1]
    rightPoint = X[rightPointIdx, 1]
    contacts = array([inHandIdxs[leftPointIdx], inHandIdxs[rightPointIdx]], dtype = "int")
    
    # find all contact points within the range specified by contactWidth
    lX, lN = point_cloud.FilterWorkspace(
      [(-1, 1), (leftPoint, leftPoint + contactWidth), (-1, 1)], X, N)
    rX, rN = point_cloud.FilterWorkspace(
      [(-1, 1), (rightPoint - contactWidth, rightPoint), (-1, 1)], X, N)
    
    # form all pairs of contacts
    nl = lX.shape[0]
    nr = rX.shape[0]
    lX = tile(lX, (nr, 1))
    lN = tile(lN, (nr, 1))
    rX = repeat(rX, nl, axis = 0)
    rN = repeat(rN, nl, axis = 0)
    
    # draw a line between contacts
    lines = lX - rX
    lineLengths = norm(lines, axis = 1)
    
    # line is ambiguous when contact points are indistinct: in this case set arbitrarily.
    pointGraspIdx = lineLengths == 0
    lines[pointGraspIdx, :] = array([0.0, 1.0, 0.0])
    lineLengths[pointGraspIdx] = 1.0
    
    # normalize lines
    lines = lines / tile(reshape(lineLengths, (lines.shape[0], 1)), (1, 3))
    
    # the grasp is antipodal iff the line between both contacts is in both friction cones.
    # we assume antipodality if any contact pair is antipodal
    isAntipodal =  logical_and( \
      sum(+lines * lN, axis = 1) >= cosHalfAngleOfFrictionCone, \
      sum(-lines * rN, axis = 1) >= cosHalfAngleOfFrictionCone).any()
    
    return isAntipodal, contacts
    
  def IsHandUnderTable(self, bTh):
    '''TODO'''
  
    bX = point_cloud.Transform(bTh, self.externalHandPoints)
    return (bX[:, 2] < self.GetTableHeight()).any()
    
  def IsPlacementStable(self, cloud, faceDistTol):
    '''Checks if the cloud is approximately in a stable placement.
    - Input cloud: Point cloud (nx3 numpy array) at the placement pose w.r.t. world coordinates.
    - Input faceDistTol: For the object to be stable, a facet in the convex hull of the object must
      be perfectly aligned with the horizontal surface. We relax this slightly by requiring each
      point of the facet to be within distance faceDistTol of the table. This roughly accounts for
      noise in the point cloud and/or movement of the object after it is placed.
    - Returns isStable: True if the object is placed stably and False otherwise.
    '''
    
    # 0. Input checking
    
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("cloud must be 3D.")
    
    if not isinstance(faceDistTol, float):
      raise Exception("Expected faceDistTol to have type float but has type {}.".format(
        type(faceDistTol)))
    
    if faceDistTol < 0:
      raise Exception("faceDistTol must be non-negative but was {}.".format(faceDistTol))
      
    if cloud.size == 0:
      return True # vacuously true
    
    # 1. Compute convex hull and center of mass of the object.
    
    hull = ConvexHull(cloud)
    triangles = hull.simplices
    center = mean(cloud, axis = 0)
    
    # 2. Find the triangle that is intersected by the vector from CoM in the direction of gravity.
    # (See intersect3D_RayTriangle at http://geomalgorithms.com/a06-_intersect-2.html.)
    
    t0 = cloud[triangles[:, 0], :]
    t1 = cloud[triangles[:, 1], :]
    t2 = cloud[triangles[:, 2], :]
    
    u = t1 - t0
    v = t2 - t0
    n = cross(u, v)
    ray = array([0, 0, -1], dtype = "float")
    a = sum(n * (t0 - center), axis = 1)
    b = sum(n * ray, axis = 1)
    
    rayInterceptsTrianglePlane = logical_and(b != 0, a * b >= 0)
    t0 = t0[rayInterceptsTrianglePlane]
    t1 = t1[rayInterceptsTrianglePlane]
    t2 = t2[rayInterceptsTrianglePlane]
    u = u[rayInterceptsTrianglePlane]
    v = v[rayInterceptsTrianglePlane]
    a = a[rayInterceptsTrianglePlane]
    b = b[rayInterceptsTrianglePlane]
    
    r = a / b
    intersectPoint = center + tile(reshape(r, (r.size, 1)), (1, 3)) * ray
    
    uu = sum(u * u, axis = 1)
    uv = sum(u * v, axis = 1)
    vv = sum(v * v, axis = 1)
    w = intersectPoint - t0
    wu = sum(w * u, axis = 1)
    wv = sum(w * v, axis = 1)
    D = uv * uv - uu * vv
    
    s = (uv * wv - vv * wu) / D
    t = (uv * wu - uu * wv) / D
    inTriangle = logical_and(logical_and(s >= 0, t >= 0), s + t <= 1)
    
    if sum(inTriangle) == 0:
      print("Warning: no triangle intersected by ray from CoM along gravity.")
      return False
    
    p0 = t0[inTriangle, :]
    p1 = t1[inTriangle, :]
    p2 = t2[inTriangle, :]
    
    # 3. If intersecting face is entirely in contact with the table, the placement is stable.
    # (If multiple intersections, check all, as the object could fall to either facet.)
    
    idx = (abs(p0[:, 2] - self.tableHeight) < faceDistTol) * \
          (abs(p1[:, 2] - self.tableHeight) < faceDistTol) * \
          (abs(p2[:, 2] - self.tableHeight) < faceDistTol)
    
    # visualize face that intersects gravity
    #self.PlotCloud(vstack([p0, p1, p2]))
    
    # return result
    return idx.all()
      
  def GenerateKinBody(self, extents, name):
    '''Creates an OpenRAVE primitive object of the given primitive shape and dimensions.
    
    - Input extents: Either [diameter] for spheres, [diameter, height] for cylinders, or
      [length, width, height] for boxes.
    - Input name: A unique name (string) to assign to the object. Note the name must be unique
      in the current scene, or the object may become impossible to clear from the scene.
    - Returns body: An OpenRAVE KinBody object. The object is added at the center of the scene.
    '''
    
    # input checking
    if len(extents) < 1 or len(extents) > 3:
      raise Exception("Expected length of extents to be 1, 2, or 3; got {}.".format(len(extents)))
    
    if not isinstance(name, str):
      raise Exception("Expected str for name; got {}.".format(name))
    
    allRaveBodies = self.env.GetBodies()
    for raveBody in allRaveBodies:
      if raveBody.GetName() == name:
        raise Exception("Name {} not unique.".format(name))
        
    # specify KinBody geometry
    geomInfo = openravepy.KinBody.Link.GeometryInfo()
    
    if len(extents) == 1:
      geomInfo._type = openravepy.KinBody.Link.GeomType.Sphere
      geomInfo._vGeomData = [extents[0] / 2.0]
    elif len(extents) == 2:
      geomInfo._type = openravepy.KinBody.Link.GeomType.Cylinder
      geomInfo._vGeomData = [extents[0] / 2.0, extents[1]]
    elif len(extents) == 3:
      geomInfo._type = openravepy.KinBody.Link.GeomType.Box
      geomInfo._vGeomData = [extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0]
    else:
      raise Exception("Extents should have only length 1, 2, or 3.")
    
    geomInfo._vDiffuseColor = self.colors[randint(len(self.colors))]
    body = openravepy.RaveCreateKinBody(self.env, "")
    body.InitFromGeometries([geomInfo])
    body.SetName(name)
    body.extents = extents
    self.env.Add(body, True)
    self.objects.append(body)
    return body
    
  def Load3DNetObject(self, fileName, scale):
    '''TODO'''

    kinBodyName = re.findall("/[^/]*.ply$", fileName)[0][1:-4]
    self.env.Load(fileName, {"scalegeometry":str(scale)})
    body = self.env.GetKinBody(kinBodyName)
    self.objects.append(body)
    return body
    
  def LoadShapeNetObject(self, fileName, height):
    '''TODO'''

    kinBodyName = re.findall("/[^/]*.obj$", fileName)[0][1:-4]
    
    # load object to determine scale
    self.env.Load(fileName)
    body = self.env.GetKinBody(kinBodyName)
    scale = height / (2 * body.ComputeAABB().extents()[1])
    self.env.Remove(body)
    
    # load object again at the desired scale
    self.env.Load(fileName, {"scalegeometry":str(scale)})
    body = self.env.GetKinBody(kinBodyName)
    self.objects.append(body)
    
    # set to canonical transform
    R = eye(4)
    R[0:3, 0] = array([1,  0, 0])
    R[0:3, 1] = array([0,  0, 1])
    R[0:3, 2] = array([0, -1, 0])
    T = eye(4)
    T[0:3, 3] = -body.ComputeAABB().pos()
    T = dot(R, T)
    body.SetTransform(T)
    
    return body, scale, T
    
  def PrintString(self, string, wait):
    '''Prints a string, either blocking for user feedback or not.
    - Input string: The string to print to the terminal.
    - Input wait: If True, waits for user to press key. Otherwise, does not wait.
    - Returns None.
    '''
    
    if not wait:
      print(string)
    else:
      raw_input(string)
    
  def RemoveObjectAtRandom(self):
    '''TODO'''
    
    if len(self.unplacedObjects) == 0: return
    self.RemoveUnplacedObject(self.unplacedObjects[randint(len(self.unplacedObjects))])
    
  def RemoveObjectNearestAnotherObject(self):
    '''TODO'''
    
    # input checking
    nObjects = len(self.unplacedObjects)
    if nObjects == 0: return
    
    if nObjects == 1:
      self.RemoveUnplacedObject(self.unplacedObjects[0])
      return
    
    # find the centroid of each object
    centers = self.GetObjectCentroids(self.unplacedObjects)
    
    # find the bottle nearest to any other object
    nearestObject = None; nearestDistance = float('inf')
    for i in xrange(nObjects):
      
      distance = []
      for j in xrange(nObjects):
        if i == j: continue
        distance.append(sum(power(centers[i, :] - centers[j, :], 2)))
      distance = min(distance)
      
      if distance < nearestDistance:
        nearestObject = self.unplacedObjects[i]
        nearestDistance = distance
    
    # remove the nearest object
    self.RemoveUnplacedObject(nearestObject)
    
  def RemoveObjectNearestCloud(self, cloud):
    '''TODO'''
    
    # input checking
    nObjects = len(self.unplacedObjects)
    if nObjects == 0: return
      
    if nObjects == 1:
      self.RemoveUnplacedObject(self.unplacedObjects[0])
      return
    
    # find distances between cloud centroid and object centroids
    estimatedCenter = mean(cloud, axis = 0)
    estimatedCenter = tile(estimatedCenter, (nObjects, 1))
    actualCenters = self.GetObjectCentroids(self.unplacedObjects)
    
    # remove nearest object
    nearestIdx = argmin(sum(power(estimatedCenter - actualCenters, 2), axis = 1))
    self.RemoveUnplacedObject(self.unplacedObjects[nearestIdx])

  def RemoveUnplacedObject(self, obj):
    '''TODO'''
    
    self.unplacedObjects.remove(obj)
    self.objects.remove(obj)
    self.RemoveObjectSet([obj])
    
  def ResetScene(self):
    '''TODO'''
    
    self.RemoveObjectSet(self.objects)
    self.objects = []
    self.placedObjects = []
    self.unplacedObjects = []