'''A class for regrasp planning. Extends ideas in "Regrasping" by Tournassoud et al., 1987.'''

# python
from multiprocessing.connection import Client
# scipy
import matplotlib
from numpy.linalg import norm
from scipy.spatial import cKDTree
from numpy.random import randint, uniform
from numpy import arccos, argmin, arange, array, ascontiguousarray, concatenate, cos, cross, \
  dot, eye, exp, hstack, isnan, log, logical_and, logical_not, max, mean, minimum, nonzero, ones, \
  pi, power, repeat, reshape, sum, tile, unique, vstack, where, zeros
# openrave
import openravepy
# self
import point_cloud
import c_extensions
import hand_descriptor
from planner import Planner
from hand_descriptor import HandDescriptor
  
class PlannerRegrasp(Planner):

  def __init__(self, env, temporaryPlacePosition, nGraspSamples, halfAngleOfGraspFrictionCone,
    nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor, stepCost, segmUncertCostFactor,
    compUncertCostFactor, graspUncertCostFactor, placeUncertCostFactor, antipodalCostFactor,
    insignificantCost, maxSearchDepth, maxIterations, minPointsPerSegment, preGraspOffset):
    '''Initialize a regrasp planner.
    
    - Input env: EnvironmentPickPlace instance for visualization and collision checking.
    - Input temporaryPlacePosition: Obstacle-free position on the table where objects can safely
      be temporary placed for regrasping. 3-element numpy array.
    - Input nGraspSamples: Number of grasps to sample on each planning iteration.
    - Input halfAngleOfGraspFrictionCone: 0.5 * the solid angle (in radians) of the friction cone of
      between object surfaces and the gripper. Smaller values are more conservative, but will result
      in fewer grasp samples and thus longer planning times, longer plans, or fewer solutions.
    - Input nTempPlaceSamples: Number of temporary placements to sample on each planning iteration. 
    - Input nGoalPlaceSamples: Number of goal placements to sample on each planning iteration.
    - Input taskCostFactor: Factor for the costs provied by the arrangement/task planner.
    - Input stepCost: The (scalar) plan cost for any grasp/place.
    - Input segmUncertCostFactor: Cost factor for uncertainty in object segmentation.
    - Input compUncertCostFactor: Cost factor for uncertainty in shape completion.
    - Input graspUncertCostFactor: Cost factor for grasping points with high completion uncertainty.
    - Input antipodalCostFactor: Cost factor penalizing large angles between the normal force and
      closing direction.
    - Input placeUncertCostFactor: Cost factor for placing on points with high completion uncertainty.
    - Input insignificantCost: A cost, if added to the final plan cost, is considered insignificant.
      This makes it possible to exit early if a plan cost exceeds a lower bound on the cost.
    - Input maxSearchDepth: Maximum plan length before restarting the search with additional grasp
      and place samples. This is an upper bound for plan length: if set too low, sometimes no
      solution will be found when solutions exist. Setting too low will also result in quickly
      sampling additional grasps / places. This value should (typically) be at least 4, i.e.,
      at least 1 temporary placement and regrasp.
    - Input maxIterations: Maximum number of sampling iterations for the regrasp planner. The
      regrasp planner may exit early if a lower bound on the cost is met.
    - Input minPointsPerSegment: Needed by Planner base class: a segment must have this many points
      to be considered an object.
    - Input preGraspOffset: Needed by Planner base class: pre-grasp is offset by this distance in
      negative approach direction.
    '''
    
    Planner.__init__(self, env, minPointsPerSegment, preGraspOffset)
    
    # input checking
    I = (int, long)
    if not isinstance(nGraspSamples, I) or not isinstance(nTempPlaceSamples, I) or \
     not isinstance(nGoalPlaceSamples, I) or not isinstance(maxSearchDepth, I) or \
     not isinstance(maxIterations, I):
       raise Exception("An integer parameter is not integer-valued.")
    
    if nGraspSamples < 0:
      raise Exception("nGraspSamples be non-negative.")
    
    if halfAngleOfGraspFrictionCone < 0 or halfAngleOfGraspFrictionCone > pi / 2.0:
      raise Exception("halfAngleOfGraspFrictionCone must be in [0, pi/2]")
    
    if nTempPlaceSamples < 0 or nGoalPlaceSamples < 0:
      raise Exception("Place parameters must be non-negative.")      
    
    if taskCostFactor < 0 or stepCost < 0 or segmUncertCostFactor < 0 or compUncertCostFactor < 0 \
      or graspUncertCostFactor < 0 or placeUncertCostFactor < 0 or antipodalCostFactor < 0 or \
      insignificantCost < 0: raise Exception("Cost factors all assumed to be be non-negative.")
      
    if maxSearchDepth < 0 or maxIterations < 0:
      raise Exception("Search parameters must be non-negative.")
    
    # grasp parameters
    self.nGraspSamples = nGraspSamples
    self.halfAngleOfGraspFrictionCone = halfAngleOfGraspFrictionCone
    self.cosHalfAngleOfGraspFrictionCone = cos(halfAngleOfGraspFrictionCone)
    self.nominalDescriptor = HandDescriptor(eye(4)) # for hand dimensions
    self.handWidth = self.nominalDescriptor.width
    self.handDepth = self.nominalDescriptor.depth
    self.squareHandWidth = self.nominalDescriptor.width**2 # for quick grasp feasibility checking
    
    # place parameters
    self.nTempPlaceSamples = nTempPlaceSamples
    self.nGoalPlaceSamples = nGoalPlaceSamples
    self.temporaryPlacePosition = temporaryPlacePosition
    
    # cost factors
    self.taskCostFactor = taskCostFactor
    self.stepCost = stepCost
    self.segmUncertCostFactor = segmUncertCostFactor
    self.compUncertCostFactor = compUncertCostFactor
    self.graspCostFactor = graspUncertCostFactor
    self.placeCostFactor = placeUncertCostFactor
    self.antipodalCostFactor = antipodalCostFactor
    self.insignificantCost = insignificantCost
    
    # search parameters
    self.maxSearchDepth = maxSearchDepth  
    self.maxIterations= maxIterations
    
    # internal variables (pre-sample grasps)
    self.preSampledClouds = []
    self.preSampledPairs = []
    self.preSampledBinormals = []
    self.preSampledGrasps = []
    
    # internal variables (continue planning)
    self.contTargObjs = []
    self.contUniqueTargObjs = []
    
  def Continue(self):
    '''TODO'''
    
    # load data saved from Plan
    targObjs = self.contTargObjs
    uniqueTargObjs = self.contUniqueTargObjs    
    
    # continue each worker
    connections = []
    for i in xrange(len(uniqueTargObjs)):
      
      # send continue message to each worker
      message = ["regrasp-continue"]
      try:
        print("Continuing regrasp planning worker {}.".format(i))
        connection = Client(('localhost', 7000 + i))
      except:
        for connection in connections: connection.close()
        raise Exception("Unable to connect to worker {}.".format(i))
      connection.send(message)
      connections.append(connection)
    
    # wait for result
    pickPlaces = []; goalIdxs = []; costs = []; workerIdxs = []
    done = zeros(len(connections), dtype = 'bool')
    i = -1
    
    while not done.all():
      
      # check the next connection
      i = 0 if i + 1 == len(connections) else i + 1      
      
      # ignore connections that are done
      if done[i]: continue
      
      # if there is a result waiting, receive it
      if connections[i].poll(0.1):        
        message = connections[i].recv()
        connections[i].close()
        purpose = message[0]
        data = message[1:]
        if purpose == "regrasp-continue":
          pickPlaces.append(data[0])
          goalIdxs.append(data[1])
          costs.append(data[2])
          workerIdxs.append(i)
          done[i] = True
          print("Worker {} completed with cost {}.".format(i, costs[-1]))
        else:
          print("Warning: message had purpose {}".format(message[0]))
        continue
      
      # send cancel if there is a solution
      if len(costs) > 0:
        try: connections[i].send(["regrasp-continue-cancel", min(costs)])
        except: print("Warning: failed to send cancel to worker {}.".format(i))
    
    # return best result
    bestIdx = argmin(costs)
    print("Worker {} had best cost {} with object {}.".format( \
      workerIdxs[bestIdx], costs[bestIdx], uniqueTargObjs[workerIdxs[bestIdx]]))
      
    targObjIdx = uniqueTargObjs[workerIdxs[bestIdx]]
    goalIdx = where(targObjIdx == targObjs)[0][goalIdxs[bestIdx]]

    return pickPlaces[bestIdx], goalIdx, targObjIdx
    
  def ContinueWorker(self, connection):
    '''TODO'''
    
    # Load data saved from previous call to PlanWorker.
    
    cloud = self.contCloud
    normals = self.contNormals
    graspContacts = self.contGraspContacts
    graspBinormals = self.contGraspBinormals
    pointCosts = self.contPointCosts
    triangleNormals = self.contTriangleNormals
    supportingTriangles = self.contSupportingTriangles
    cloudCenter = self.contCloudCenter
    originalGoalPlaces = self.contOriginalGoalPlaces
    goalPlaces = self.contGoalPlaces
    goalCosts = self.contGoalCosts
    graspPlaceTable = self.contGraspPlaceTable
    grasps = self.contGrasps
    graspCosts = self.contGraspCosts
    tempPlaces = self.contTempPlaces
    placeCosts = self.contPlaceCosts
    obstacleCloudAtStart = self.contObstacleCloudAtStart
    obstacleCloudAtGoal = self.contObstacleCloudAtGoal
    
    # Search for plans.
    
    plan = []; cost = float('inf')
    
    nReachableStart = sum(graspPlaceTable[:, 0] > 0) if graspPlaceTable.shape[1] > 0 else 0
    nReachableGoal = sum(graspPlaceTable[:, 1 + len(tempPlaces):] > 0) \
      if graspPlaceTable.shape[1] > 0 else 0
    
    for iteration in xrange(self.maxIterations):
      
      if len(plan) > 0: break # stop as soon as a plan is found
      
      if connection.poll():
        message = connection.recv()
        if message[0] == "regrasp-continue-cancel":
          print("Received cancel.")
          self.SaveContinuationData(plan, cloud, normals, graspContacts, graspBinormals, pointCosts,
            triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces,
            goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
            obstacleCloudAtStart, obstacleCloudAtGoal)
          return [], None, float('inf')
      
      graspsNew, pairsNew = self.SampleGrasps(cloud, normals, graspContacts, graspBinormals)
      graspCostsNew = self.GetGraspCosts(cloud, normals, pointCosts, pairsNew)
        
      if nReachableStart > 0 and nReachableGoal > 0:
        tempPlacesNew, tempPlacesNewTriangles = self.SamplePlaces(cloud, supportingTriangles,
          triangleNormals, cloudCenter)
        tempPlaceCostsNew = self.GetPlaceCosts(pointCosts, tempPlacesNewTriangles)
      else:
        tempPlacesNew = []
        tempPlaceCostsNew = zeros(0)
      
      goalPlacesNew = originalGoalPlaces[len(goalPlaces) : len(goalPlaces) + \
        self.nGoalPlaceSamples] if nReachableStart > 0 else []
      goalPlaceCostsNew = goalCosts[len(goalPlaces) : len(goalPlaces) + self.nGoalPlaceSamples] \
        if nReachableStart > 0 else zeros(0)
      
      graspPlaceTable, placeTypes, grasps, graspCosts, tempPlaces, goalPlaces, placeCosts, \
        nReachableStart, nReachableTemp, nReachableGoal = self.UpdateGraspPlaceTable(
        graspPlaceTable, grasps, graspsNew, graspCosts, graspCostsNew, tempPlaces, tempPlacesNew,
        goalPlaces, goalPlacesNew, placeCosts, tempPlaceCostsNew, goalPlaceCostsNew,
        obstacleCloudAtStart, obstacleCloudAtGoal)
      if nReachableStart == 0 or nReachableGoal == 0: continue
      
      plan, cost = self.SearchGraspPlaceGraph(graspPlaceTable, placeTypes, graspCosts, placeCosts)
      print("Iteration: {}. Regrasp plan: {}. Cost: {}.".format(iteration, plan, cost))
    
    # Return result
    self.SaveContinuationData(plan, cloud, normals, graspContacts, graspBinormals, pointCosts,
      triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces, goalCosts,
      graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts, obstacleCloudAtStart,
      obstacleCloudAtGoal)
    pickPlaces, goalIdx = self.PlanToTrajectory(plan, grasps, tempPlaces, goalPlaces)
    return pickPlaces, goalIdx, cost
    
  def GetAntipodalPairsOfPoints(self, cloud, normals):
    '''Given a point cloud of an object with corresponding surface normals, computes "antipodal"
       pairs of points. See spatial antipodal grasps in "A Mathematical Introduction to Robotic
       Manipulation" by Murray, Li, and Sastry. This is a preprocessing step for sampling hand poses
       for antipodal grasps.
      
    - Input cloud: nx3 numpy array of points.
    - Input normals: nx3 numpy array of surface normals (assumes norm(normals, axis = 1) = 1).
    - Returns pairs: List of pairs of indices into cloud, indicating pairs of points which can form
      antipodal grasps, assuming the gripper is not in collision and makes contact with these points.
    - Returns binormals: List of unit vectors in direction of line connecting the corresponding
      point pairs. These are candidate hand closing directions.
    '''
    
    # Input checking    
    
    if cloud.shape[0] != normals.shape[0]:
      raise Exception("Cloud with {} points does not match {} normals.".format(
        cloud.shape[0], normals.shape[0]))
    
    if cloud.shape[1] != 3 or normals.shape[1] != 3:
      raise Exception("Cloud or normals not 3D.")
    
    # Identify antipodal pairs of points.
    # A point pair is antipodal if the line drawn between them lies within both friction cones, i.e.
    # is within some threshold angle of the object's surface normal.
    
    # (Consider all pairs of points, in both directions.)
    i = reshape(tile(arange(cloud.shape[0]), cloud.shape[0]), (cloud.shape[0]**2, 1))
    j = reshape(repeat(arange(cloud.shape[0]), cloud.shape[0]), (cloud.shape[0]**2, 1))
    pairs = hstack((i, j))    
    
    # (Vectorize calculation of binormals for each pair of points in the cloud.)
    X = tile(cloud, (cloud.shape[0], 1))
    Y = repeat(cloud, cloud.shape[0], axis = 0)
    binormals = X - Y
    # remove pairs not containing distinct points ---
    binormalsLen = norm(binormals, axis = 1)
    uniquePointsInPair = binormalsLen > 0
    binormals = binormals[uniquePointsInPair]
    binormalsLen = binormalsLen[uniquePointsInPair]
    pairs = pairs[uniquePointsInPair]
    # --
    binormals = binormals / tile(reshape(binormalsLen, (binormals.shape[0], 1)), (1, 3))
    
    # (Vectorize antipodal check.)
    X = tile(normals, (normals.shape[0], 1))
    Y = repeat(normals, normals.shape[0], axis = 0)
    X = X[uniquePointsInPair]
    Y = Y[uniquePointsInPair]
    isAntipodal = logical_and(\
      sum(+binormals * X, axis = 1) >= self.cosHalfAngleOfGraspFrictionCone, \
      sum(-binormals * Y, axis = 1) >= self.cosHalfAngleOfGraspFrictionCone)
    pairs = pairs[isAntipodal]; binormals = binormals[isAntipodal]
    #print("Found {} antipodal pairs of points.".format(len(pairs)))
    
    # Filter pairs which are too wide for the gripper.
    
    fits = sum(power(cloud[pairs[:, 0]] - cloud[pairs[:, 1]], 2), axis = 1) < self.squareHandWidth
    pairs = pairs[fits]
    binormals = binormals[fits]    
    #print("{} pairs fit in gripper.".format(len(pairs)))
    
    return pairs, binormals
    
  def GetCostLowerBound(self, cloud, normals, pointCosts, pointPairs, goalPlaceCosts):
    '''Computes a lower bound on the cost acheivable by a regrasp plan.
    
    - Input cloud: nx3 numpy array of points representing the object.
    - Input normals: nx3 numpy array of surface normals at the corresponding points in cloud
      (assumed to be normalized).
    - Input pointCosts: n-element numpy array of costs associated with each point.
    - Input pointPairs: mx2 integer numpy array, where, each row indexes a pair of points in cloud
      that could potentially form an antipodal grasp.
    - Input goalPlaceCosts: k-element list of task-relevant costs for each goal placement.
    - Returns: A (scalar) lower bound on the regrasp plan cost. (The regrasp plan cost is described
      in detail in the Notebook, proposal, paper, etc.)
    '''
    
    if pointPairs.size == 0:
      return float('inf') # no grasps possible
      
    if len(goalPlaceCosts) == 0:
      return float('inf') # no goal placements possible
    
    # there must be at least 2 steps, 2 grasps, and 1 goal
    graspCosts = self.GetGraspCosts(cloud, normals, pointCosts, pointPairs)
    return 2.0 * self.stepCost + 2.0 * min(graspCosts) + min(goalPlaceCosts) + self.insignificantCost
    
  def GetGraspCosts(self, cloud, normals, pointCosts, pointPairs):
    '''Given a pair of contact points for parallel-jaw grasps, find the step cost for each grasp.
    
    - Input cloud: nx3 numpy array of points representing the object.
    - Input normals: nx3 numpy array of surface normals at the corresponding points in cloud
      (assumed to be normalized).
    - Input pointCosts: n-element numpy array of costs associated with each point.
    - Input pointPairs: mx2 integer numpy array, where, each row indexes a pair of points in cloud
      that could potentially form an antipodal grasp.
    - Returns graspCosts: m-element numpy array, which is the estimated negative half log
      probability of grasp success. (Half because the search algorithm will penalize each grasp
      twice, once for the pick and once for the place.)
    '''
    
    # input checking
    if len(pointPairs) == 0:
      return zeros(0)
      
    if cloud.shape[0] != normals.shape[0]:
      raise Exception("Cloud with {} points does not match {} normals.".format(
        cloud.shape[0], normals.shape[0]))
        
    if cloud.shape[0] != pointCosts.shape[0]:
      raise Exception("Cloud with {} points does not match {} point costs.".format(
        cloud.shape[0], pointCosts.shape[0]))
        
    if cloud.shape[1] != 3 or normals.shape[1] != 3:
      raise Exception("Cloud or normals not 3D.")
      
    if max(pointPairs) >= cloud.shape[0]:
      raise Exception("cloud has {} points, but pointPairs indices go up to {}.".format(
        cloud.shape[0], max(pointPairs)))
    
    # uncertainty at contact points
    uncertaintyCost = 0.5 * self.graspCostFactor * sum(pointCosts[pointPairs], axis = 1)
    
    # antipodal cost
    if self.antipodalCostFactor > 0:
      
      binormals = cloud[pointPairs[:, 0], :] - cloud[pointPairs[:, 1], :]
      binormalsLen = norm(binormals, axis = 1)
      binormals /= tile(reshape(binormalsLen, (binormals.shape[0], 1)), (1, 3))
      N0 = normals[pointPairs[:, 0], :]; N1 = normals[pointPairs[:, 1], :]
      
      mu0 = arccos(minimum(sum(+binormals * N0, axis = 1), 1.0))
      mu1 = arccos(minimum(sum(-binormals * N1, axis = 1), 1.0))
      
      # With this cost, the friction cone is the random variable, centered on the given surface
      # normal with half angle uniformly distributed between 0 and pi.
      #cost0 = -log(1.0 - mu0, 1.0)) / pi)
      #cost1 = -log(1.0 - mu1, 1.0)) / pi)
      
      # With this cost, the angle between the normal and binormal is normally distributed (but
      # truncated between 0 and pi), and we can compute the probability it is outside the friction
      # cone using the truncated normal CDF.
      cost0 = -log(self.TruncatedNormalCdf(self.halfAngleOfGraspFrictionCone, mu0, 0.13, 0.00, pi))
      cost1 = -log(self.TruncatedNormalCdf(self.halfAngleOfGraspFrictionCone, mu1, 0.13, 0.00, pi))
      
      antipodalCost = 0.5 * self.antipodalCostFactor * (cost0 + cost1)
      
    else:
      antipodalCost = zeros(len(pointPairs))
    
    return uncertaintyCost + antipodalCost
    
  def GetPlaceCosts(self, pointCosts, triangles):
    '''Given points that contact the horizontal surface for a temporary placement, determines the
       cost of the placement.
    
    - Input pointCosts: n-element numpy array with costs for each point in the point nx3 point cloud.
    - Input triangles: kx3 integer numpy array where each row indexes a triple in point costs,
      corresponding to the contact points of a placement. (Output by SamplePlacements.)
    - Returns placeCosts: k-element numpy array with the cost for each triple of contact points,
      i.e., for each temporary placement to be evaluated.
    '''
    
    if triangles.shape[1] != 3:
      raise Exception("Triangles must be 3D.")
      
    if max(triangles) >= pointCosts.shape[0]:
      raise Exception("pointCosts has {} elements, but triangle indices go up to {}.".format(
        pointCosts.shape[0], max(triangles)))
    
    return 0.5 * self.placeCostFactor * sum(pointCosts[triangles], axis = 1)

  def GetPointCosts(self, segmClouds, compClouds, segmCorrectProbs, compCorrectProbs):
    '''Associates a cost with each point in the completed point clouds.
    
    - Input segmClouds: Result of object instance segmentation, i.e., a list of point clouds
      (m_i x 3 numpy arrays), one for each object.
    - Input compClouds: Shape completions for each object, i.e., list of n_i x 3 numpy arrays.
    - Input segmCorrectProbs: List of m_i, per-point segmentation correctness probabilities.
    - Input compCorrectProbs: List of n_i, per-point completion correctness probabilities.
    - Returns pointCosts: n_i non-negative costs for each point of each object.
    '''
    
    # input checking
    if len(compClouds) != len(segmClouds) or len(segmCorrectProbs) != len(segmClouds) or \
      len(compCorrectProbs) != len(segmClouds):
        raise Exception("Inconsistent number of objects.")
    
    pointCosts = []
    for i in xrange(len(segmClouds)):
      
      # associate segmentation probabilities with completion probabilities
      
      # (method 1: use average segmentation correctness probability)
      #associatedSegmCorrectProbs = ones(compCorrectProbs[i].size) * mean(segmCorrectProbs[i])
      
      # (method 2: use nearest segmentation correctness probability)
      tree = cKDTree(segmClouds[i])
      _, idx = tree.query(compClouds[i])
      associatedSegmCorrectProbs = segmCorrectProbs[i][idx]
      
      # convert correctness probabilities to costs
      pointCosts.append(-self.segmUncertCostFactor * log(associatedSegmCorrectProbs) - \
        self.compUncertCostFactor * log(compCorrectProbs[i]))
        
    # debugging
    #self.env.PlotCloud(concatenate(compClouds), matplotlib.cm.viridis(exp(-concatenate(pointCosts))))
    #raw_input("Contact point probabilities.")
      
    return pointCosts
  
  def Plan(self, segmClouds, segmCorrectProbs, compClouds, compCorrectProbs, normals, goalPoses,
    goalCosts, targObjs, obstaclesAtStart, obstaclesAtGoal):
    '''Runs PlanWorker for each object in parellel and returns the lowest-cost regrasp plan found.
    
    - Input segmClouds: Segmentation for each object.
    - Input segmCorrectProbs: Per-point segmentation correctness probability (for each object).
    - Input compClouds: Completed point cloud for each object.
    - Input compCorrectProbs: Per-point segmentation correctness probabiltiy (for each object).
    - Input normals: Surface normals for each completed point cloud.
    - Input goalPoses: A list of goal placements (transform of the object from its current pose).
    - Input goalCosts: The task cost associated with each goal pose.
    - Input targObjs: The index of the object associated with each goal pose.
    - Input obstaclesAtStart: Point cloud of obstacles at object's current location (for each goal
      pose). Assumed same for identical target objects.
    - Input obstaclesAtGoal: Point cloud of obstacles at object's goal location (for each goal
      pose). Assumed same for identical target objects.
    - Returns pickPlaces: Gripper poses for each pick/place in the regrasp plan.
    - Returns goalIdx: The index (in goalPoses, goalCosts, or targObjs) of the goal placement.
    - Returns targObjIdx: The index (in compClouds) of the object selected for manipulation.
    '''
    
    # input checking
    nObjects = len(segmCorrectProbs); nGoals = len(goalPoses)
    if len(segmCorrectProbs) != nObjects or len(compClouds) != nObjects or \
      len(compCorrectProbs) != nObjects or len(normals) != nObjects:
        raise Exception("Inconsistent number of objects.")
    if len(goalCosts) != nGoals or len(targObjs) != nGoals or len(obstaclesAtStart) != nGoals or \
      len(obstaclesAtGoal) != nGoals:
        raise Exception("Inconsistent number of goals.")
    if isnan(goalCosts).any(): raise Exception("Goal costs must be non-NaN.")
    if min(goalCosts) < 0: raise Exception("Goal costs must be non-negative.")
    
    # compute "point costs" and incorporate task cost factor
    pointCosts = self.GetPointCosts(segmClouds, compClouds, segmCorrectProbs, compCorrectProbs)
    for i in xrange(len(goalCosts)):
      goalCosts[i] *= self.taskCostFactor
    
    # get a list of objects to plan for
    uniqueTargObjs = unique(targObjs)
    
    # create a worker for each object
    connections = []; bestPossibleCosts = zeros(len(uniqueTargObjs))
    for i, targObj in enumerate(uniqueTargObjs):
      
      # pre-sample grasps if this hasn't been done yet
      hasPreSampled = False
      for j, preSampledCloud in enumerate(self.preSampledClouds):
        if compClouds[targObj] is preSampledCloud:
          hasPreSampled = True
          break
      if not hasPreSampled:
        self.PreSampleGraspsOnObjects(
          [compClouds[targObj]], [normals[targObj]], [pointCosts[targObj]], 0)
        j = 0
      
      # fetch pre-sampled grasps
      grasps = self.preSampledGrasps[j]
      graspContacts = self.preSampledPairs[j]
      graspBinormals = self.preSampledBinormals[j]
      graspCosts = self.preSampledGraspCosts[j]
      
      # get all of the goals and costs for this object
      finalPoses = [goalPoses[k] for k in xrange(len(targObjs)) if targObj == targObjs[k]]
      finalCosts = [goalCosts[k] for k in xrange(len(targObjs)) if targObj == targObjs[k]]
      startObstacles = [obstaclesAtStart[k] for k in xrange(len(targObjs)) if targObj == targObjs[k]]
      finalObstacles = [obstaclesAtGoal[k] for k in xrange(len(targObjs)) if targObj == targObjs[k]]
      finalCosts = array(finalCosts)
      
      # find the best possible cost for this object      
      bestPossibleCosts[i] = self.GetCostLowerBound(compClouds[targObj], normals[targObj],
        pointCosts[targObj], graspContacts, finalCosts)
      
      # send relevant data to each worker
      message = ["regrasp", compClouds[targObj], normals[targObj], pointCosts[targObj], finalPoses,
        finalCosts, startObstacles[0], finalObstacles[0], grasps, graspCosts, graspContacts,
        graspBinormals, bestPossibleCosts[i]]
      try:
        print("Starting regrasp planning worker {} with goal costs {}.".format(i, finalCosts))
        connection = Client(('localhost', 7000 + i))
      except:
        for connection in connections: connection.close()
        raise Exception("Unable to connect to worker {}.".format(i))
      connection.send(message)
      connections.append(connection)
    
    # determine when to cancel workers
    print("Best possible plan cost is {}.".format(min(bestPossibleCosts) - self.insignificantCost))
    
    # wait for result
    pickPlaces = []; goalIdxs = []; costs = []; workerIdxs = []
    done = zeros(len(connections), dtype = 'bool')
    i = -1
    
    while not done.all():
      
      # check the next connection
      i = 0 if i + 1 == len(connections) else i + 1      
      
      # ignore connections that are done
      if done[i]: continue
      
      # if there is a result waiting, receive it
      if connections[i].poll(0.1):        
        message = connections[i].recv()
        connections[i].close()
        purpose = message[0]
        data = message[1:]
        if purpose == "regrasp":
          pickPlaces.append(data[0])
          goalIdxs.append(data[1])
          costs.append(data[2])
          workerIdxs.append(i)
          done[i] = True
          # the lower bound for this worker is no longer achievable
          bestPossibleCosts[i] = data[2] + self.insignificantCost
          print("Worker {} completed with cost {}.".format(i, costs[-1]))
        else:
          print("Warning: message had purpose {}".format(message[0]))
        continue
      
      # send cancel if there is a solution and the best cost is acheived
      if len(costs) > 0 and min(costs) <= min(bestPossibleCosts):
        try: connections[i].send(["regrasp-cancel", min(costs)])
        except: print("Warning: failed to send cancel to worker {}.".format(i))
        
    # save data for continuation
    self.contTargObjs = targObjs
    self.contUniqueTargObjs = uniqueTargObjs
    
    # return best result
    bestIdx = argmin(costs)
    print("Worker {} had best cost {} with object {}.".format( \
      workerIdxs[bestIdx], costs[bestIdx], uniqueTargObjs[workerIdxs[bestIdx]]))
      
    targObjIdx = uniqueTargObjs[workerIdxs[bestIdx]]
    goalIdx = where(targObjIdx == targObjs)[0][goalIdxs[bestIdx]]

    return pickPlaces[bestIdx], goalIdx, targObjIdx
    
  def PlanWorker(self, cloud, normals, pointCosts, goalPlaces, goalCosts, obstacleCloudAtStart,
    obstacleCloudAtGoal, graspsNew, graspCostsNew, graspContacts, graspBinormals, bestPossibleCost,
    connection):
    '''Searches for a minimum-cost regrasp plan for the given target object.
    
    - Input cloud: Point cloud (nx3 numpy array) representing the object.
    - Input normals: Surface normals (nx3 numpy array, assumed normalized) for the object.
    - Input pointCosts: n-element array of contact costs associated with each point.
    - Input goalPlaces: List of m placement goals, 4x4 transforms of object from it's current pose.
    - Input goalCosts: Costs associated with each placement goal (m, non-negative elements).
    - Input obstacleCloudAtStart: Point cloud representing obstacles at the object's initial location.
    - Input obstacleCloudAtGoal: Point cloud representing obstacles at the object's goal location.
    - Input graspsNew: List of k initially sampled grasps (4x4 transforms). (More are sampled as
      this function executes.)
    - Input graspCostsNew: Costs for each of the k grasps.
    - Input graspContacts: Pairs of indices into cloud for contact points for each of the k grasps.
    - Input graspBinormals: Closing direction for each of the k grasps.
    - Input bestPossibleCost: A lower bound on the cost that can be achieved by this 
    - Input connection: An open Python Connection, allowing sending messages to and receiving
      messages from the parent process. (PlanWorker is invoked when a message from the parent
      process is received. The parent process waits in Plan for PlanWorker to finish.)
    - Returns pickPlaces: List of gripper poses (4x4 homogeneous transforms), relative to the world
      frame, for each grasp and place. The list is of even length. Returns None if no plan is found.
    - Returns goalIdx: Index into goalPlaces that was ultimatly used for placing the object. Returns
      None if no regrasp plan is found.
    - Returns cost: The cost of the regrasp plan. Inf if no plan is found.
    '''
    
    # Input checking
      
    if cloud.shape[0] != normals.shape[0]:
      raise Exception("Cloud and normals must have the same number of points.")
      
    if cloud.shape[1] != 3 or normals.shape[1] != 3:
      raise Exception("Cloud and normals must be 3D.")
    
    if pointCosts.size != cloud.shape[0]:
      raise Exception("Must have the same number of point costs as points.")
    
    if isnan(pointCosts).any():
      raise Exception("Point costs must not be NaN.")
      
    if len(pointCosts) > 0 and min(pointCosts) < 0:
      raise Exception("Point costs must be non-negative.")
    
    if len(goalPlaces) != len(goalCosts):
      raise Exception("Mismatch in size of goalPlaces and goalCosts.")
    
    if isnan(goalCosts).any():
      raise Exception("Goal costs must not be NaN.")
      
    if len(goalCosts) > 0 and min(goalCosts) < 0:
      raise Exception("Goal costs must be non-negative.")
      
    if obstacleCloudAtStart.shape[1] != 3 or obstacleCloudAtGoal.shape[1] != 3:
      raise Exception("Obstacle clouds must be 3D.")
    
    if len(graspsNew) != len(graspCostsNew):
      raise Exception("Mismatch in size of graspsNew and graspCostsNew.")
      
    if isnan(graspCostsNew).any():
      raise Exception("Grasp costs must not be NaN.")
    
    if min(graspCostsNew) < 0:
      raise Exception("Grasp costs must be non-negative.")
      
    if graspContacts.shape[0] != graspBinormals.shape[0]:
      raise Exception("Mismatch in size between graspContacts and graspBinormals.")
      
    if graspContacts.shape[1] != 2:
      raise Exception("graspContacts must be 2D.")
    
    if graspBinormals.shape[1] != 3:
      raise Exception("graspBinormals must be 3D.")
      
    # Initialize regrasp planning graph and other variables.
    
    grasps = []
    graspCosts = zeros(0)
    tempPlaces = []
    originalGoalPlaces = goalPlaces
    goalPlaces = []
    placeCosts = zeros(0)
    graspPlaceTable = zeros((0, 1), dtype = "int32", order = "C")
    
    # Initialize place sampler.
    
    supportingTriangles, triangleNormals, cloudCenter = self.GetSupportSurfaces(cloud)
    
    if len(supportingTriangles) == 0:
      print("No supporting faces found! No regrasp plan found.")
      self.SaveContinuationData([], cloud, normals, graspContacts, graspBinormals, pointCosts,
        triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces,
        goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
        obstacleCloudAtStart, obstacleCloudAtGoal)
      return [], None, float('inf')
      
    # Check for early exit.
      
    if len(graspContacts) == 0:
      print("No antipodal contacts found! No regrasp plan found.")
      self.SaveContinuationData([], cloud, normals, graspContacts, graspBinormals, pointCosts,
        triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces,
        goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
        obstacleCloudAtStart, obstacleCloudAtGoal)
      return [], None, float('inf')
    
    # Search for plans.
    
    plan = []; cost = float('inf')
    nReachableStart = 1; nReachableGoal = 1; notFirstLoop = False  
    
    for iteration in xrange(self.maxIterations):
      
      if cost <= bestPossibleCost: break
      
      if connection.poll():
        message = connection.recv()
        if message[0] == "regrasp-cancel":
          print("Received cancel.")
          self.SaveContinuationData([], cloud, normals, graspContacts, graspBinormals, pointCosts,
            triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces,
            goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts, 
            obstacleCloudAtStart, obstacleCloudAtGoal)
          return [], None, float('inf')
      
      #print("Sampling grasps and places.")
      if notFirstLoop:
        graspsNew, pairsNew = self.SampleGrasps(cloud, normals, graspContacts, graspBinormals)
        graspCostsNew = self.GetGraspCosts(cloud, normals, pointCosts, pairsNew)
      else:
        notFirstLoop = True
        
      if nReachableStart > 0 and nReachableGoal > 0:
        tempPlacesNew, tempPlacesNewTriangles = self.SamplePlaces(cloud, supportingTriangles,
          triangleNormals, cloudCenter)
        tempPlaceCostsNew = self.GetPlaceCosts(pointCosts, tempPlacesNewTriangles)
      else:
        tempPlacesNew = []
        tempPlaceCostsNew = zeros(0)
      
      goalPlacesNew = originalGoalPlaces[len(goalPlaces) : len(goalPlaces) + \
        self.nGoalPlaceSamples] if nReachableStart > 0 else []
      goalPlaceCostsNew = goalCosts[len(goalPlaces) : len(goalPlaces) + self.nGoalPlaceSamples] \
        if nReachableStart > 0 else zeros(0)
      
      #print("Updating grasp-place table.")
      graspPlaceTable, placeTypes, grasps, graspCosts, tempPlaces, goalPlaces, placeCosts, \
        nReachableStart, nReachableTemp, nReachableGoal = self.UpdateGraspPlaceTable(
        graspPlaceTable, grasps, graspsNew, graspCosts, graspCostsNew, tempPlaces, tempPlacesNew,
        goalPlaces, goalPlacesNew, placeCosts, tempPlaceCostsNew, goalPlaceCostsNew,
        obstacleCloudAtStart, obstacleCloudAtGoal)
      if nReachableStart == 0 or nReachableGoal == 0: continue
      
      #print("Searching grasp-place graph.")
      plan, cost = self.SearchGraspPlaceGraph(graspPlaceTable, placeTypes, graspCosts, placeCosts)
      print("Iteration: {}. Regrasp plan: {}. Cost: {}.".format(iteration, plan, cost))
    
    # Return result
    
    pickPlaces, goalIdx = self.PlanToTrajectory(plan, grasps, tempPlaces, goalPlaces)
    
    self.SaveContinuationData(plan, cloud, normals, graspContacts, graspBinormals, pointCosts,
      triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces, goalCosts,
      graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts, obstacleCloudAtStart,
      obstacleCloudAtGoal)    
    
    return pickPlaces, goalIdx, cost
    
  def PlanToTrajectory(self, plan, grasps, tempPlaces, goalPlaces):
    '''Converts a plan into a sequence of gripper poses.
    
    - Input plan: List of indices, where the 1st index is into grasps and the 2nd index is into 
      [start, tempPlaces, goalPlaces].
    - Input grasps: List of homogeneous transforms representing antipodal grasps on the object in
      the base frame at the object's starting configuration.
    - Input tempPlaces: List of homogeneous transforms representing the object at a temporary place.
    - Input goalPlaces: List of homogeneous transforms representing the object's goal placements.
    - Returns pickPlaces: List of transforms representing the gripper in the base frame at
      alternating grasp and ungrasp (i.e. place) configurations (starting with grasping the object
      at its present configuration). 
    - Returns goalIdx: Index into goalPlaces for the goal configuration the plan ends with.
    '''
    
    pickPlaces = []
    goalIdx = None
    for item in plan:
      graspIdx = item[0]
      placeIdx = item[1]
      if placeIdx == 0:
        placePose = eye(4)
      elif placeIdx <= len(tempPlaces):
        placePose = tempPlaces[placeIdx - 1]
      else:
        goalIdx = placeIdx - 1 - len(tempPlaces)
        placePose = goalPlaces[goalIdx]
      pickPlaces.append(dot(placePose, grasps[graspIdx]))
      
    return pickPlaces, goalIdx
    
  def PreSampleGraspsOnObjects(self, clouds, normals, pointCosts, nSamples, visualize = False):
    '''Samples grasps on each object to be used next time Plan is called.
    
    - Input clouds: A point cloud for each object to sample grasps on.
    - Input normals: Object surface normals for each cloud (assumed to be normalized).
    - Input pointCosts: Per-point contact costs for each cloud.
    - Input nSamples: Number of grasps to randomly sample.
    - Input visualize: If True, shows the grasps found in the viewer for each object.
    - Returns preGraspSamples: Grasps for each object sampled.
    '''
    
    # input checking
    
    if len(normals) != len(clouds) or len(pointCosts) != len(clouds):
      raise Exception("Inconsistent number of objects.")
    
    # sample grasps for each object    
    self.preSampledClouds = clouds; self.preSampledPairs = []; self.preSampledBinormals = []
    self.preSampledGrasps = []; self.preSampledGraspCosts = []
    
    for i in xrange(len(clouds)):
      
      pairs, binormals = self.GetAntipodalPairsOfPoints(clouds[i], normals[i])
      tmpSamples = self.nGraspSamples; self.nGraspSamples = nSamples
      grasps, pairsForGrasps = self.SampleGrasps(clouds[i], normals[i], pairs, binormals)
      self.nGraspSamples = tmpSamples
      graspCosts = self.GetGraspCosts(clouds[i], normals[i], pointCosts[i], pairsForGrasps)
      
      self.preSampledPairs.append(pairs)
      self.preSampledBinormals.append(binormals)
      self.preSampledGrasps.append(grasps)
      self.preSampledGraspCosts.append(graspCosts)
      
      if visualize:
        self.env.PlotDescriptors(hand_descriptor.DescriptorsFromPoses(grasps),
          matplotlib.cm.viridis(exp(-graspCosts)))
        raw_input("Grasps for object {}.".format(i))
        
    return self.preSampledGrasps
    
  def SampleGrasps(self, cloud, normals, pairs, binormals):
    '''Samples antipodal grasps. See spatial antipodal grasps in "A Mathematical Introduction to
       Robotic Manipulation" by Murray, Li, and Sastry.
    
    - Input cloud: nx3 numpy array of points.
    - Input normals: nx3 numpy array of surface normal directions for each point in cloud, rows
      assumed to be normalized.
    - Input pairs: mx2 integer numpy where each row is a pair of indices into cloud indicating
      antipodal contact points. Computed in GetAntipodalPairsOfPoints.
    - Input binormals: mx3 numpy array of hand closing directions, corresponding to pairs, from
      GetAntipodalPairsOfPoints. Assumed to be normalized.
    - Returns grasps: List of k, 4x4 numpy matrices which are homogeneous transforms indicating
      randomly sampled grasps in the world frame. Grasps (approximately) make contact with antipodal
      pairs and are guaranteed to be collision-free (checked with the full gripper model) with the
      target object. Antipodal pairs, orientation about the binormal, and offset in approach
      direction are sampled uniformly at random.
    - Returns pairs: kx2 integer numpy array where each row is a pair of indices into the point
      cloud, indicating contacts, for each corresponding grasp.
    '''
    
    # Check inputs.
    
    if len(pairs) == 0:
      return [], zeros((0, 2), dtype = "int32")
      
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("Point cloud not 3D.")
      
    if len(normals.shape) != 2 or normals.shape[1] != 3:
      raise Exception("Surface normals not 3D.")
      
    if normals.shape[0] != cloud.shape[0]:
      raise Exception("Cloud has {} points but normals has {}.".format(
        cloud.shape[0], normals.shape[0]))
    
    if pairs.shape[1] != 2:
      raise Exception("Pairs not 2D.")    
    
    if max(pairs) >= cloud.shape[0]:
      raise Exception("Cloud has {} points, but indices in pairs go up to {}.".format(
        cloud.shape[0], max(pairs)))
    
    if len(binormals.shape) != 2 or binormals.shape[1] != 3:
      raise Exception("Binormals not 3D.")    
    
    if binormals.shape[0] != pairs.shape[0]:
      raise Exception("Binormals has {} elements but pairs has {} elements.".format(
        binormals.shape[0], pairs.shape[0]))
    
    # Sample hand poses from the remaining pairs of points.
    # The binormal (hand closing direction) is now fixed. Center x, y on the pair of points. 
    # This leaves 2 free parameters: the approach direction and approach (z) offset.
    
    theta = uniform(0, 2 * pi, self.nGraspSamples)
    offset = uniform(-self.handDepth / 2.0, self.handDepth / 2.0, self.nGraspSamples)
    idx = randint(0, pairs.shape[0], self.nGraspSamples)
    
    pairs = pairs[idx]
    binormals = binormals[idx]
    
    grasps = []
    for i in xrange(self.nGraspSamples):
      
      approach = self.GetOrthogonalUnitVector(binormals[i])
      center = (cloud[pairs[i][0]] + cloud[pairs[i][1]]) / 2.0
      T0 = hand_descriptor.PoseFromApproachBinormalCenter(approach, binormals[i], center)
      R = openravepy.rotationMatrixFromAxisAngle(binormals[i], theta[i])[0:3, 0:3]
      
      T = eye(4)
      T[0:3, 0:3] = dot(R, T0[0:3, 0:3])
      T[0:3, 3] = center + offset[i] * T[0:3, 2]
      grasps.append(T)
      
    # Filter grasp samples whos actual contacts are not antipodal. This is important because we
    # implicitly assumed the pair along the binormal were contacts. Also, determine which pair of
    # points is truly in contact with the grasp.
    
    keepGrasps = []; contactPairs = []
    for i, grasp in enumerate(grasps):
     
     isAntipodal, pair = self.env.IsAntipodalGrasp(grasp, cloud, normals,
       self.cosHalfAngleOfGraspFrictionCone, self.env.graspContactWidth)
     
     if isAntipodal:
       keepGrasps.append(grasp)
       contactPairs.append(pair)
       
    grasps = keepGrasps
    pairs = array(contactPairs)
    
    # Filter grasp samples that collide with the object.
    collisionFree = logical_not(self.env.CheckHandObjectCollision(grasps, cloud))
    grasps = [grasps[i] for i in xrange(len(grasps)) if collisionFree[i]]
    pairs = pairs[collisionFree]
    
    # Return result
    return grasps, pairs
    
  def SamplePlaces(self, cloud, supportingTriangles, triangleNormals, cloudCenter):
    '''Randomly transforms for the point cloud where the object will rest stably on the supporting
       surface.
    
    - Input cloud: Point cloud of the object -- an nx3 numpy array.
    - Input supportingTriangles: Indices into cloud indicating facets on which the object will rest
      stably -- an mx3 integer numpy array. Each row corresponds to one facet. Can be computed by
      Planner.GetSupportSurfaces.
    - Input triangleNormals: Outward-pointing normals for supportTriangles -- mx3 numpy array with
      normalized rows. Can be comptued by Planner.GetSupportSurfaces.
    - Input cloudCenter: The mean point of the cloud -- 3-element numpy array. Can be computed by
      Planner.GetSupportSurfaces.
    - Returns places: List of k transforms which move the cloud from its current pose to the stable
      placement, relative to the world frame -- list of 4x4 numpy arrays.
    - Returns supportingTriangles: The support facets from supportingTriangles used -- kxn numpy
      integer array.
    '''
    
    # input checking
    if supportingTriangles.shape[0] != triangleNormals.shape[0]:
      raise Exception("supportingTriangles and triangleNormals must have same number of elements.")
    
    if cloud.shape[1] != 3 or supportingTriangles.shape[1] != 3 or triangleNormals.shape[1] != 3 \
      or cloudCenter.size != 3: raise Exception("Inputs must be 3D.")
    
    if supportingTriangles.shape[0] == 0:
      return [], zeros((0, 3), dtype = "int")
    
    # randomly sample rotation of object about gravity and supporting face
    theta = uniform(0, 2 * pi, self.nTempPlaceSamples)
    idx = randint(0, supportingTriangles.shape[0], self.nTempPlaceSamples)    
    
    places = []
    for i in xrange(self.nTempPlaceSamples):
      
      # attach a coordinate system to the cloud such that z is -triangleNormal
      bTo = eye(4)
      bTo[0:3, 3] = cloudCenter
      bTo[0:3, 2] = -triangleNormals[idx[i]]
      R = openravepy.matrixFromAxisAngle(bTo[0:3, 2], theta[i])[0:3, 0:3]
      bTo[0:3, 1] = dot(R, self.GetOrthogonalUnitVector(bTo[0:3, 2]))
      bTo[0:3, 0] = cross(bTo[0:3, 1], bTo[0:3, 2])
      
      # check the cloud in its new orientation
      oTb = point_cloud.InverseTransform(bTo)
      cloudRotated = point_cloud.Transform(oTb, cloud)
      
      # determine translation
      bToo = eye(4)
      bToo[0, 3] = self.temporaryPlacePosition[0]
      bToo[1, 3] = self.temporaryPlacePosition[1]
      bToo[2, 3] = self.temporaryPlacePosition[2] - min(cloudRotated[:, 2])
      places.append(dot(bToo, oTb))
    
    # visualization
    '''for i, place in enumerate(places):
      self.env.PlotCloud(point_cloud.Transform(place, cloud))
      raw_input("Showing {}th stable placement.".format(i))'''
    
    return places, supportingTriangles[idx]
    
  def SaveContinuationData(self, plan, cloud, normals, graspContacts, graspBinormals, pointCosts,
    triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces, goalPlaces, goalCosts,
    graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts, obstacleCloudAtStart,
    obstacleCloudAtGoal):
    '''TODO'''
    
    # remove grasps that are part of the current plan 
    keepMask = ones(len(grasps), dtype = "bool")
    for i, step in enumerate(plan):
      keepMask[step[0]] = False
    graspPlaceTable = graspPlaceTable[keepMask, :]
    grasps = [grasps[i] for i in xrange(len(grasps)) if keepMask[i]]
    graspCosts = graspCosts[keepMask]
    
    # save 
    self.contCloud = cloud
    self.contNormals = normals
    self.contGraspContacts = graspContacts
    self.contGraspBinormals = graspBinormals
    self.contPointCosts = pointCosts
    self.contTriangleNormals = triangleNormals
    self.contSupportingTriangles = supportingTriangles
    self.contCloudCenter = cloudCenter
    self.contOriginalGoalPlaces = originalGoalPlaces
    self.contGoalPlaces = goalPlaces
    self.contGoalCosts = goalCosts
    self.contGraspPlaceTable = graspPlaceTable
    self.contGrasps = grasps
    self.contGraspCosts = graspCosts
    self.contTempPlaces = tempPlaces
    self.contPlaceCosts = placeCosts
    self.contObstacleCloudAtStart = obstacleCloudAtStart
    self.contObstacleCloudAtGoal = obstacleCloudAtGoal
    
  def SearchGraspPlaceGraph(self, graspPlaceTable, placeTypes, graspCosts, placeCosts):
    '''Calls the C++ routine for performing A* search on the grasp-place table.
    
    - Input graspPlaceTable: Grasp-place table, with values in {-1, 0, 1}, assumed to be C-order of
      type int32.
    - Input placeTypes: Place types, with values in {0, 1, 2} assumed to be C-order of type int32.
    - Input graspCosts: The cost for selecting each grasp. A 1D numpy array of length
      graspPlaceTable.shape[0].
    - Input placeCosts: The cost for selecting each place. a 1D numpy array of length
      graspPlaceTable.shape[1].
    - Returns planList: List of pairs, [(grasp, place)_1, ..., (grasp, place)_n]
    - Returns cost: The total plan cost, equal to len(plan) * self.stepCost + goalCosts[goalIdx],
      where goalIdx corresponds to the placement selected.
    '''
    
    # input checking
    
    if placeTypes.size != graspPlaceTable.shape[1] or graspCosts.size != graspPlaceTable.shape[0] \
      or placeCosts.size != graspPlaceTable.shape[1]:
        raise Exception("GraspPlace table has shape {}, but there are {} place types, {} grasp " + \
          "costs, and {} place costs.".format(graspPlaceTable.shape, placeTypes.size,
          graspCosts.size, placeCosts.size))
    
    graspCosts = ascontiguousarray(graspCosts, dtype = "float32")
    placeCosts = ascontiguousarray(placeCosts, dtype = "float32")
    
    # call C++ routine    
    plan = -1 * ones(2 * self.maxSearchDepth, dtype = "int32", order = "C")    
    cost = c_extensions.SearchGraspPlaceGraph(graspPlaceTable.shape[0], graspPlaceTable.shape[1],
      graspPlaceTable, placeTypes, graspCosts, placeCosts, self.stepCost, self.maxSearchDepth, plan)
    
    # make plan 2D
    planList = []
    for i in xrange(self.maxSearchDepth):
      if plan[2 * i + 0] < 0 or plan[2 * i + 1] < 0: break
      planList.append((plan[2 * i + 0], plan[2 * i + 1]))
    
    # return result
    return planList, cost
    
  def UpdateGraspPlaceTable(self, graspPlaceTable, grasps, graspsNew, graspCosts, graspCostsNew,
    tempPlaces, tempPlacesNew, goalPlaces, goalPlacesNew, placeCosts, tempPlaceCostsNew,
    goalPlaceCostsNew, obstacleCloudAtStart, obstacleCloudAtGoal):
    '''Expands the grasp-place table with new grasp and place samples and checks feasibility of
       each essential unchecked grasp-place combination.
    
    - Input graspPlaceTable: mxn numpy array, where the ith row corresponds to grasp i, the jth
      column corresponds to place j, and the value is 1 if the grasp i is feasible at place j, -1 if
      infeasible, and 0 if unknown.
    - Input grasps: m-element list of grasps (4x4 transforms) corresponding to rows of graspPlaceTable.
    - Input graspsNew: m'-element list of grasps corresponding to grasps to be added.
    - Input graspCosts: m-element numpy array of costs associated with each grasp in grasps.
    - Input graspCostsNew: m'-element numpy array of costs associated with each grasp in newGrasps.
    - Input tempPlaces: List of existing temporary places (4x4 transforms) corresponding to columns
      1 to 1 + len(tempPlaces) of graspPlaceTable.
    - Input tempPlacesNew: List of temporary places to add as middle columns of graspPlaceTable.
    - Input goalPlaces: List of existing goal places corresponding to the last len(goalPlaces)
      columns of graspPlaceTable. n = 1 + len(goalPlaces) + len(tempPlaces).
    - Input goalPlacesNew: List of goal places to add to graspPlaceTable.
      n' = len(tempPlacesNew) + len(goalPlacesNew).
    - Input placeCosts: n-element array of costs associated with existing places.
    - Input tempPlaceCostsNew: Costs associated with tempPlacesNew.
    - Input goalPlaceCostsNew: Costs associated with goalPlacesNew.
    - Input obstacleCloudAtStart: Point cloud (nx3 numpy array) representing obstacles at the start.
    - Input obstacleCloudAtGoal: Point cloud representing obstacles at all goal places.
    - Returns graspPlaceTable: graspPlaceTable of shape (m+m')x(n+n') augmented with new grasps and
      new places, checked for feasiblilty. (Some feasibility checks can be skipped if no grasps are
      available at the start or any goal.)
    - Returns placeTypes: n+n'-element array with the type of each place: 0 at position j means the
      jth place/column is a start place, 1 is a temporary place, and 2 is a goal place.
    - Returns grasps: m+m'-element list of grasps corresponding to graspPlaceTable.
    - Returns graspCosts: m+m'-element list of costs assicated with each grasp in grasps.
    - Returns tempPlaces: List of places at the regrasp area.
    - Returns goalPlaces: List of places at the goal area.
    - Returns placeCosts: n+n'-element list of costs associated with each place.
    - Returns nReachableStart: The number of grasps reachable at the start pose of the object.
    - Returns nReachableTemp: The number of grasps reachable at temporary placements.
    - Returns nReachableGoal: The number of grasps reachable at goal placements.
    '''
    
    # Input checking.

    if len(grasps) != graspPlaceTable.shape[0]:
      raise Exception("Input {} grasps but graspPlaceTable has {} rows.".format(\
        len(grasps), graspPlaceTable.shape[0]))
        
    if len(grasps) != graspCosts.size:
      raise Exception("Input {} grasps but have {} grasp costs.".format(\
        len(grasps), graspCosts.size))
    
    if len(graspsNew) != graspCostsNew.size:
      raise Exception("Input {} new grasps but have {} new grasp costs.".format(\
        len(graspsNew), graspCostsNew.size))
        
    if 1 + len(tempPlaces) + len(goalPlaces) != graspPlaceTable.shape[1]:
      raise Exception("Input {} places but graspPlaceTable has {} columns.".format(\
        1 + len(tempPlaces) + len(goalPlaces), graspPlaceTable.shape[1]))
        
    if len(tempPlacesNew) != tempPlaceCostsNew.size:
      raise Exception("Input {} new temp. places but have {} new temp. place costs.".format(\
        len(tempPlacesNew), tempPlaceCostsNew.size))
        
    if len(goalPlacesNew) != goalPlaceCostsNew.size:
      raise Exception("Input {} new goal places but have {} new goal costs.".format(\
        len(goalPlacesNew), goalPlaceCostsNew.size))
        
    if obstacleCloudAtStart.shape[1] != 3:
      raise Exception("Obstacle cloud at start not 3D.")
      
    if obstacleCloudAtGoal.shape[1] != 3:
      raise Exception("Obstacle cloud at goal not 3D.")
      
    if isnan(graspCosts).any(): raise Exception("graspCosts has NaN.")
    if isnan(graspCostsNew).any(): raise Exception("graspCostsNew has NaN.")
    if isnan(placeCosts).any(): raise Exception("placeCosts has NaN.")
    if isnan(tempPlaceCostsNew).any(): raise Exception("tempPlaceCostsNew has NaN.")
    if isnan(goalPlaceCostsNew).any(): raise Exception("goalPlaceCostsNew has NaN.") 
    
    # Semantics of grasp-place table: ======================
    # Rows: Places in order [start, temporary, goal]
    # Columns: [grasps].T
    # Values: (0) not checked, (-1) invalid, and (1) valid
    # ======================================================

    # Add a row for each new grasp.
    newRows = zeros((len(graspsNew), graspPlaceTable.shape[1]), dtype = "int32", order = "C")
    graspPlaceTable = vstack([graspPlaceTable, newRows])
    # Add a column for each new temporary place.
    newColumns = zeros((graspPlaceTable.shape[0], len(tempPlacesNew)), dtype = "int32", order = "C")
    graspPlaceTable = hstack([graspPlaceTable[:, :len(tempPlaces) + 1], newColumns,
      graspPlaceTable[:, len(tempPlaces) + 1:]])
    # Add a column for each new final place.
    newColumns = zeros((graspPlaceTable.shape[0], len(goalPlacesNew)), dtype = "int32", order = "C")
    graspPlaceTable = hstack([graspPlaceTable, newColumns])
    
    # Concatenate old and new grasp and place lists.
    placeCosts = concatenate([zeros(1), placeCosts[1:len(tempPlaces) + 1], tempPlaceCostsNew,
      placeCosts[len(tempPlaces) + 1:], goalPlaceCostsNew]) # must be before next 2 lines
    grasps += graspsNew; tempPlaces += tempPlacesNew; goalPlaces += goalPlacesNew
    graspCosts = concatenate([graspCosts, graspCostsNew])
    
    # Record place type for each column: (0) start, (1) temporary, and (2) goal.
    placeTypes = zeros(1 + len(tempPlaces) + len(goalPlaces), dtype = 'int32', order = "C")
    placeTypes[1:len(tempPlaces) + 1] = 1
    placeTypes[len(tempPlaces) + 1:] = 2 
    
    # Mark grasps as being valid / invalid.
    
    # find where checks need made
    row, col = nonzero(graspPlaceTable == 0)
    row = reshape(row, (row.size, 1))
    col = reshape(col, (col.size, 1))
    checkIdxs = hstack([row, col])
    
    # find grasp in place frame
    posesStart = []; posesTemp = []; posesGoal = []
    idxsStart = []; idxsTemp = []; idxsGoal = []
    for idx in checkIdxs:
      idx = tuple(idx)
      graspIdx = idx[0]
      placeIdx = idx[1]
      grasp = grasps[graspIdx]
      if placeTypes[placeIdx] == 0:
        posesStart.append(grasp)
        idxsStart.append(idx)
      elif placeTypes[placeIdx] == 1:
        place = tempPlaces[placeIdx - 1]
        posesTemp.append(dot(place, grasp))
        idxsTemp.append(idx)
      else:
        place = goalPlaces[placeIdx - len(tempPlaces) - 1]
        posesGoal.append(dot(place, grasp))
        idxsGoal.append(idx)
    
    # check reachability at start
    reachableStart = self.env.CheckReachability(
      posesStart, self.GetPreGrasps(posesStart), True, obstacleCloudAtStart)
    for i, idx in enumerate(idxsStart):
      graspPlaceTable[idx] = 1 if reachableStart[i] else -1
    nReachableStart = sum(graspPlaceTable[:, 0] > 0)
    if nReachableStart == 0:
      print("Found 0 start grasps.")
      return graspPlaceTable, placeTypes, grasps, graspCosts, tempPlaces, goalPlaces, placeCosts, \
        0, 0, 0
    
    # check reachability at goal
    reachableGoal = self.env.CheckReachability(
      posesGoal, self.GetPreGrasps(posesGoal), True, obstacleCloudAtGoal)
    for i, idx in enumerate(idxsGoal):
      graspPlaceTable[idx] = 1 if reachableGoal[i] else -1
    nReachableGoal = sum(graspPlaceTable[:, 1 + len(tempPlaces):] > 0)
    if nReachableGoal == 0:
      print("Found {} start and 0 goal grasps.".format(nReachableStart))
      return graspPlaceTable, placeTypes, grasps, graspCosts, tempPlaces, goalPlaces, placeCosts, \
        nReachableStart, 0, 0
    
    # check reachability at temporary
    reachableTemp = self.env.CheckReachability(posesTemp, self.GetPreGrasps(posesTemp), True)
    for i, idx in enumerate(idxsTemp):
      graspPlaceTable[idx] = 1 if reachableTemp[i] else -1
    nReachableTemp = sum(graspPlaceTable[:, 1 : len(tempPlaces) + 1] > 0)
    
    # Return result
    
    print("Found {} start, {} temporary, and {} goal grasps.".format(
      nReachableStart, nReachableTemp, nReachableGoal))
    
    return graspPlaceTable, placeTypes, grasps, graspCosts, tempPlaces, goalPlaces, placeCosts, \
      nReachableStart, nReachableTemp, nReachableGoal