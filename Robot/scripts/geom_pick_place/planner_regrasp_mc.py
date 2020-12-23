'''Regrasp planner which uses the Monte Carlo (MC) sampling cost for grasps and places.'''

# python
from multiprocessing.connection import Client
# scipy
import matplotlib
from numpy.linalg import norm
from scipy.special import erfinv
from numpy.random import choice, normal
from numpy import argmax, argmin, arange, array, copy, exp, isnan, log, max, min, mean, ones, \
  power, logical_and, repeat, reshape, sqrt, sum, tile, unique, where, zeros
# self
import point_cloud
import hand_descriptor
from planner_regrasp import PlannerRegrasp
  
class PlannerRegraspMC(PlannerRegrasp):

  def __init__(self, env, temporaryPlacePosition, nGraspSamples, graspFrictionCone,
    nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor, stepCost, segmUncertCostFactor,
    compUncertCostFactor, graspUncertCostFactor, placeUncertCostFactor, antipodalCostFactor,
    insignificantCost, maxSearchDepth, maxIterations, minPointsPerSegment, preGraspOffset):
    '''Initialize MC regrasp planner. Parameters are the same as PlannerRegrasp.'''
    
    PlannerRegrasp.__init__(self, env, temporaryPlacePosition, nGraspSamples, graspFrictionCone,
      nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor, stepCost, segmUncertCostFactor,
      compUncertCostFactor, graspUncertCostFactor, placeUncertCostFactor, antipodalCostFactor,
      insignificantCost, maxSearchDepth, maxIterations, minPointsPerSegment, preGraspOffset)
      
  def ContinueWorker(self, connection):
    '''TODO'''
    
    # Load data saved from previous call to PlanWorker.
    
    cloud = self.contCloud
    normals = self.contNormals
    graspContacts = self.contGraspContacts
    graspBinormals = self.contGraspBinormals
    cloudSamples = self.contCloudSamples
    normalSamples = self.contNormalSamples
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
          self.SaveContinuationData(plan, cloud, normals, graspContacts, graspBinormals,
            cloudSamples, normalSamples, triangleNormals, supportingTriangles, cloudCenter,
            originalGoalPlaces, goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts,
            tempPlaces, placeCosts, obstacleCloudAtStart, obstacleCloudAtGoal)
          return [], None, float('inf')
      
      graspsNew, _ = self.SampleGrasps(cloud, normals, graspContacts, graspBinormals)
      graspCostsNew = self.GetGraspCosts(graspsNew, cloudSamples, normalSamples)
        
      if nReachableStart > 0 and nReachableGoal > 0:
        tempPlacesNew, tempPlacesNewTriangles = self.SamplePlaces(cloud, supportingTriangles,
          triangleNormals, cloudCenter)
        tempPlaceCostsNew = self.GetPlaceCosts(tempPlacesNew, cloudSamples)
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
    self.SaveContinuationData(plan, cloud, normals, graspContacts, graspBinormals, cloudSamples,
      normalSamples, triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces,
      goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
      obstacleCloudAtStart, obstacleCloudAtGoal)
    pickPlaces, goalIdx = self.PlanToTrajectory(plan, grasps, tempPlaces, goalPlaces)
    return pickPlaces, goalIdx, cost
      
  def GetCostLowerBound(self, goalPlaceCosts):
    '''Computes a lower bound on the cost acheivable by a regrasp plan.
    
    - Input goalPlaceCosts: k-element list of task-relevant costs for each goal placement.
    - Returns: A (scalar) lower bound on the regrasp plan cost. (The regrasp plan cost is described
      in detail in the Notebook, proposal, paper, etc.)
    '''
    
    if goalPlaceCosts.size == 0:
      return float('inf') # no goal placements possible
    
    # there must be at least 2 steps and 1 goal
    return 2.0 * self.stepCost + min(goalPlaceCosts) + self.insignificantCost
    
  def GetGraspCosts(self, grasps, sampleClouds, sampleNormals):
    '''Find the cost of each grasp, i.e., the negative log success rate.
    
    - Input grasps: List of n grasps (4x4 homogeneous transforms) to evaluate the cost for.
    - Input sampleClouds: List of N, n_i x 3 numpy arrays, samples of the same object shape.
    - Input sampleNormals: List of N, n_i x 3 numpy arrays, surface normals, assumed normalized.
    - Returns graspCosts: n-element array, negative log (estimated) probability of grasp success.
    '''
    
    # Input checking
    
    if len(grasps) == 0:
      return zeros(0, dtype = "float")
    
    if len(sampleClouds) == 0 or self.graspCostFactor == 0:
      return zeros(len(grasps), dtype = "float") # -self.graspCostFactor * log(1) = 0
      
    if len(sampleClouds) != len(sampleNormals):
      raise Exception("Length of sampleClouds, {}, must be the same as sampleNormals, {}.".format(
        len(sampleClouds), len(sampleNormals)))
    
    # Compute grasp feasibility for each shape
    
    nFeasibleShapes = ones(len(grasps), dtype = "float")
    for i in xrange(len(sampleClouds)):
      
      # put clouds in hand reference frame
      cloudsInHandFrame = []; normalsInHandFrame = []
      for j in xrange(len(grasps)):
        hTb = point_cloud.InverseTransform(grasps[j])
        C, N = point_cloud.Transform(hTb, sampleClouds[i], sampleNormals[i])
        cloudsInHandFrame.append(C); normalsInHandFrame.append(N)
      
      # Method 1: check antipodal and collision (using simplified hand model)
      #isAntipodal = self.IsAntipodalGrasp(cloudsInHandFrame, normalsInHandFrame)
      #nFeasibleShapes += self.IsGripperCollisionFreeHeuristic(cloudsInHandFrame, isAntipodal)
      
      # Method 2: just check antipodal condition
      nFeasibleShapes += self.IsAntipodalGrasp(cloudsInHandFrame, normalsInHandFrame)
    
    graspCosts = 0.5 * self.graspCostFactor * log((1.0 + len(sampleClouds)) / nFeasibleShapes)
    
    # Visualization
    '''for i, grasp in enumerate(grasps):
      self.env.PlotDescriptors([HandDescriptor(grasp)])
      raw_input("Grasp {} has cost {} and {} feasible shapes.".format(
        i, graspCosts[i], nFeasibleShapes[i]))'''
    
    return graspCosts
    
  def GetPlaceCosts(self, places, sampleClouds):
    '''Find the cost of each place, i.e., negative log place success rate.
    
    - Input places: List of n placements (4x4 homogeneous tranforms) to test.
    - Input sampleClouds: List of N point clouds (n_i x 3), sampling shapes of the same object.
    - Returns placeCosts: n-element array, negative log (estimated) probabiltiy of place stability.
    '''
    
    # Input checking
    
    if len(places) == 0:
      return zeros(0, dtype = "float")
      
    if len(sampleClouds) == 0 or self.placeCostFactor == 0:
      return zeros(len(places), dtype = "float")
    
    # Check which shapes will be placed stably.
    
    nFeasibleShapes = ones(len(places), dtype = "float")
    for i in xrange(len(sampleClouds)):
      for j in xrange(len(places)):
        
        cloudAtPlace = point_cloud.Transform(places[j], sampleClouds[i])
        isStable = self.env.IsPlacementStable(cloudAtPlace, self.env.faceDistTol)
        nFeasibleShapes[j] += isStable
          
        # visualization
        '''self.env.PlotCloud(cloudAtPlace)
        stableString = "is" if isStable else "is not"
        raw_input("Place {} stable.".format(stableString))'''
      
    placeCosts = 0.5 * self.placeCostFactor * log((1.0 + len(sampleClouds)) / nFeasibleShapes)
    
    # Visualization
    '''for i, place in enumerate(places):
      self.env.PlotCloud(point_cloud.Transform(place, sampleClouds[0]))
      raw_input("Place {} has cost {} and {} feasible shapes.".format(
        i, placeCosts[i], nFeasibleShapes[i]))'''
    
    return placeCosts
    
  def IsAntipodalGrasp(self, cloudsInHandFrame, normalsInHandFrame):
    '''This is the same antipodal check as in EnvironmentPickPlace except the input is the cloud
       in the hand frame (instead of the grasp and the cloud in the base frame) and tests several
       clouds in batch. See spatial antipodal grasps in "A Mathematical Introduction to Robotic
       Manipulation" by Murray, Li, and Sastry.
       
    - Input cloudsInHandFrame: List of N, n_i x 3 numpy arrays, representing the point cloud w.r.t.
      the hand frame, as defined in HandDescriptor (e.g., 0 is the center of the hand, z is the
      negative approach direction, etc.)
    - Input normalsInHandFrame: List of N, n_i x 3 numpy arrays, surface normals corresponding to
      the input point clouds, assumed to be normalized across the rows and in the hand frame.
    - Returns isAntipodal: N-element bool array which is True if the corresponding shape passes the
      check and False if the corresponding shape does not pass the check.
    '''
    
    # Input checking
    
    if len(cloudsInHandFrame) != len(normalsInHandFrame):
      raise Exception("cloudsInHandFrame with length {} inconsistent with normalsInHandFrame " + \
        "with length {}.".format(len(cloudsInHandFrame), len(normalsInHandFrame)))
    
    # Is antipodal?

    isAntipodal = zeros(len(cloudsInHandFrame), dtype = "bool")
    for i in xrange(len(cloudsInHandFrame)):    
    
      # put cloud into hand reference frame
      X, N = point_cloud.FilterWorkspace(
        self.handClosingRegion, cloudsInHandFrame[i], normalsInHandFrame[i])
      if X.size == 0: continue # no points in the hand
  
      # find contact points
      leftPoint = min(X[:, 1]); rightPoint = max(X[:, 1])
      lX, lN = point_cloud.FilterWorkspace(
        [(-1, 1), (leftPoint, leftPoint + self.env.graspContactWidth), (-1, 1)], X, N)
      rX, rN = point_cloud.FilterWorkspace(
        [(-1, 1), (rightPoint - self.env.graspContactWidth, rightPoint), (-1, 1)], X, N)
      
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
      
      # the grasp is antipodal iff the line between contacts is in both friction cones
      # we assume antipodality if any contact pair is antipodal
      isAntipodal[i] = logical_and(\
        sum(+lines * lN, axis = 1) >= self.cosHalfAngleOfGraspFrictionCone, \
        sum(-lines * rN, axis = 1) >= self.cosHalfAngleOfGraspFrictionCone).any()
    
    return isAntipodal
    
  def IsGripperCollisionFreeHeuristic(self, cloudsInHandFrame, checkCollisionMask):
    '''Estimates if the grippers are in collision with the target object. Instead of using the
       entire gripper model, as in EnvironmentPickPlace, this checks if points lie inside of
       rectangular regions.
    
    - Input cloudsInHandFrame: N point clouds (n_i x 3 numpy arrays) for each shape of the same
      object to check. The points are assumed to be in the grasp reference frame, as defined in
      HandDescriptor.
    - Input checkCollisionMask: N-element bool array that, if False, collision checking is skipped
      and the output for the corresponding cloud.
    - Returns isCollisionFree: N-element bool array indicating which elements were checked and were
      collision-free.
    '''

    # input checking
    if len(cloudsInHandFrame) != checkCollisionMask.size:
      raise Exception("cloudsInHandFrame length {} inconcistent with checkCollisionMask size {}." \
        .format(len(cloudsInHandFrame), checkCollisionMask.size))
    
    # check collision
    isCollisionFree = copy(checkCollisionMask)
    for i in xrange(len(cloudsInHandFrame)):
      if not checkCollisionMask[i]: continue
      isCollisionFree[i] = not self.InCollisionWithHandElements(cloudsInHandFrame[i])
      
    return isCollisionFree
  
  def Plan(self, compClouds, normals, completionSamples, completionSampleNormals, goalPoses,
    goalCosts, targObjs, obstaclesAtStart, obstaclesAtGoal):
    '''Runs PlanWorker for each object in parellel and returns the lowest-cost plan found.
    
    - Input compClouds: Completed point cloud for each object.
    - Input normals: Surface normals for each completed point cloud.
    - Input completionSamples: A list of sample shapes (completed clouds) for each object.
    - Input completionSampleNormals: Surface normals corresponding to completionSamples.
    - Input goalPoses: A list of goal placements (transform of the object from its current frame).
    - Input goalCosts: The task cost associated with each goal pose.
    - Input targObjs: The object associated with each goal pose.
    - Input obstaclesAtStart: Point cloud of obstacles at object's current location (for each goal
      pose). Assumed same for identical target objects.
    - Input obstaclesAtGoal: Point cloud of obstacles at object's goal location (for each goal
      pose). Assumed same for identical target objects.
    - Returns pickPlaces: Gripper poses for each pick/place in the regrasp plan.
    - Returns goalIdx: The index (in goalPoses, goalCosts, or targObjs) of the goal placement.
    - Returns targObjIdx: The index (in compClouds) of the object selected for manipulation.
    '''
    
    # input checking
    if len(compClouds) != len(normals):
        raise Exception("compClouds and normals have inconsistent lengths.")
    
    if len(compClouds) != len(completionSamples):
      raise Exception("compClouds and completionSamples have inconsistent lengths.")
    
    if len(compClouds) != len(completionSampleNormals):
      raise Exception("compClouds and completionSampleNormals have inconsistent lengths.")
    
    if len(goalPoses) != len(goalCosts):
      raise Exception("goalPoses and goalCosts have inconsistent lengths.")
    
    if len(goalPoses) != len(targObjs):
      raise Exception("goalPoses and targObjs have inconsistent lengths.")
      
    if len(goalPoses) != len(obstaclesAtStart):
      raise Exception("goalPoses and obstaclesAtStart have inconsistent lengths.")
      
    if len(goalPoses) != len(obstaclesAtGoal):
      raise Exception("goalPoses and obstaclesAtGoal have inconsistent lengths.")
      
    for i in xrange(len(completionSamples)):
      
      if len(completionSamples[i]) != len(completionSampleNormals[i]):
        raise Exception("Completion samples for object {} has {} samples but normals have {}."\
        .format(i, completionSamples[i].shape[0], completionSampleNormals[i].shape[0]))
      
      for j in xrange(len(completionSamples[i])):
        
        if completionSamples[i][j].shape[0] != completionSampleNormals[i][j].shape[0]:
          raise Exception("Completion sample {} for object {} has {} points but normals has {}."\
            .format(j, i, completionSamples[i][j].shape[0], completionSampleNormals[i][j].shape[0]))
        
        if len(completionSamples[i][j].shape) != 2 or completionSamples[i][j].shape[1] != 3:
          raise Exception("Completion sample {} for object {} not 3D.".format(j, i))
        
        if len(completionSampleNormals[i][j].shape) != 2 or completionSampleNormals[i][j].shape[1] != 3:
          raise Exception("Normals sample {} for object {} not 3D.".format(j, i))
      
    if isnan(goalCosts).any():
      raise Exception("Goal costs must be non-NaN.")
      
    if min(goalCosts) < 0:
      raise Exception("Goal costs must be non-negative.")
  
    # account for task cost factor
    for i in xrange(len(goalCosts)):
      goalCosts[i] *= self.taskCostFactor
    
    # get a list of objects to plan for
    uniqueTargObjs = unique(targObjs);
    
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
        self.PreSampleGraspsOnObjects([compClouds[targObj]], [normals[targObj]],
          [completionSamples[targObj]], [completionSampleNormals[targObj]], 0)
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
      bestPossibleCosts[i] = self.GetCostLowerBound(finalCosts)
      
      # send relevant data to each worker
      message = ["regrasp", compClouds[targObj], normals[targObj], completionSamples[targObj],
        completionSampleNormals[targObj], finalPoses, finalCosts, startObstacles[0],
        finalObstacles[0], grasps, graspCosts, graspContacts, graspBinormals, bestPossibleCosts[i]]
      try:
        print("Starting regrasp planning worker {} with goal costs {}.".format(i, finalCosts))
        connection = Client(('localhost', 7000 + i))
      except:
        for connection in connections: connection.close()
        raise Exception("Unable to connect to worker {}.".format(i))
      connections.append(connection)
      connections[-1].send(message)
    
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
      
      # send cancel if there is a solution and either the best cost is acheived or there is no time
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
    
  def PlanWorker(self, cloud, normals, cloudSamples, normalSamples, goalPlaces, goalCosts,
    obstacleCloudAtStart, obstacleCloudAtGoal, graspsNew, graspCostsNew, graspContacts,
    graspBinormals, bestPossibleCost, connection):
    '''Searches for a minimum-cost regrasp plan for the given target object.
    
    - Input cloud: Point cloud (nx3 numpy array) representing the object.
    - Input normals: Surface normals (nx3 numpy array, assumed row-normalized) for the object.
    - Input cloudSamples: List of N point clouds, representing sampled shapes of the object.
    - Input normalSamples: List of N surface normal matrices corresponding to each cloud sample.
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
    - Returns configs: List of arm configurations (6 joint angles), for each item in pickPlaces,
      that are in the joint limits and and are collision free given the provided obstacles. Returns
      None if no regrasp plan is found.
    - Returns goalIdx: Index into goalPlaces that was ultimatly used for placing the object. Returns
      None if no regrasp plan is found.
    - Returns cost: The cost of the regrasp plan. Inf if no plan is found.
    '''
    
    # Input checking
      
    if cloud.shape[0] != normals.shape[0]:
      raise Exception("Cloud and normals must have the same number of points.")
      
    if cloud.shape[1] != 3 or normals.shape[1] != 3:
      raise Exception("Cloud and normals must be 3D.")
    
    if len(cloudSamples) != len(normalSamples):
      raise Exception("Must have the same number of cloudSamples and normalSamples.")
    
    if len(goalPlaces) != len(goalCosts):
      raise Exception("Mismatch in size of goalPlaces and goalCosts.")
    
    if isnan(goalCosts).any():
      raise Exception("Goal costs must not be NaN.")
      
    if min(goalCosts) < 0:
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
      self.SaveContinuationData([], cloud, normals, graspContacts, graspBinormals, cloudSamples,
        normalSamples, triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces,
        goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
        obstacleCloudAtStart, obstacleCloudAtGoal)
      return [], None, float('inf')
    
    # Check for early exit
      
    if len(graspContacts) == 0:
      print("No antipodal contacts found! No regrasp plan found.")
      self.SaveContinuationData([], cloud, normals, graspContacts, graspBinormals, cloudSamples,
        normalSamples, triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces,
        goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
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
          self.SaveContinuationData([], cloud, normals, graspContacts, graspBinormals, cloudSamples,
            normalSamples, triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces,
            goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
            obstacleCloudAtStart, obstacleCloudAtGoal)
          return [], None, float('inf')     
      
      #print("Sampling grasps and places.")
      if notFirstLoop:
        graspsNew, _ = self.SampleGrasps(cloud, normals, graspContacts, graspBinormals)
        graspCostsNew = self.GetGraspCosts(graspsNew, cloudSamples, normalSamples)
      else:
        notFirstLoop = True
        
      if nReachableStart > 0 and nReachableGoal > 0:
        tempPlacesNew, tempPlacesNewTriangles = self.SamplePlaces(cloud, supportingTriangles,
          triangleNormals, cloudCenter)
        tempPlaceCostsNew = self.GetPlaceCosts(tempPlacesNew, cloudSamples)
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
    
    self.SaveContinuationData(plan, cloud, normals, graspContacts, graspBinormals, cloudSamples,
      normalSamples, triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces,
      goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
      obstacleCloudAtStart, obstacleCloudAtGoal)
    
    return pickPlaces, goalIdx, cost
    
  def PreSampleGraspsOnObjects(self, clouds, normals, cloudSamples, normalSamples, nGraspSamples,
    visualize = False):
    '''Samples grasps on each object to be used next time Plan is called.
    
    - Input clouds: A point cloud for each object to sample grasps on.
    - Input normals: Object surface normals for each cloud (assumed to be normalized).
    - Input cloudSamples: A list of shape samples (point clouds) for each object.
    - Input normalSamples: A list of normals samples, corresponding to cloudSamples..
    - Input nGraspSamples: Number of grasps to randomly sample.
    - Input visualize: If True, shows the grasps found in the viewer for each object.
    - Returns preGraspSamples: Grasps for each object sampled.
    '''
    
    # input checking
    
    if len(normals) != len(clouds):
      raise Exception("Length of normals and clouds inconsistent.")
      
    if len(cloudSamples) != len(clouds):
      raise Exception("Length of cloudSamples and clouds inconsistent.")
      
    if len(normalSamples) != len(cloudSamples):
      raise Exception("Length of normalSamples and clouds inconsistent.")
      
    if nGraspSamples < 0 or not isinstance(nGraspSamples, (int, long)):
      raise Exception("nGraspSamples must be a positive integer.")
      
    # sample grasps for each object
    
    self.preSampledClouds = clouds; self.preSampledPairs = []; self.preSampledBinormals = []
    self.preSampledGrasps = []; self.preSampledGraspCosts = []
    
    for i in xrange(len(clouds)):
      
      pairs, binormals = self.GetAntipodalPairsOfPoints(clouds[i], normals[i])
      tmpSamples = self.nGraspSamples; self.nGraspSamples = nGraspSamples
      grasps, _ = self.SampleGrasps(clouds[i], normals[i], pairs, binormals)
      self.nGraspSamples = tmpSamples
      graspCosts = self.GetGraspCosts(grasps, cloudSamples[i], normalSamples[i])
      
      self.preSampledPairs.append(pairs)
      self.preSampledBinormals.append(binormals)
      self.preSampledGrasps.append(grasps)
      self.preSampledGraspCosts.append(graspCosts)
      
      if visualize:
        self.env.PlotDescriptors(hand_descriptor.DescriptorsFromPoses(grasps),
          matplotlib.cm.viridis(exp(-graspCosts)))
        raw_input("Grasps for object {}.".format(i))
        
    return self.preSampledGrasps
    
  def SampleShapeCompletions(self, inputCloud, segmentedClouds, segmentationDist, completionModel,
    completedClouds, completionCorrectProbabilities, nCompletionSamples, certainEquivSegmProb):
    '''Randomly samples shape completions (i.e., point clouds) given (point-wise independent)
       distibutions on object instance segmentation and completion correctness.
    
    - Input inputCloud: Point cloud of the scene (nx3 numpy array).
    - Input segmentedClouds: List of m point clouds, one for each object (from the estimated object
      instance segmentation.)
    - Input segmentationDist: An nxM matrix, where each row is a probability distribution over
      object ID (each row corresponds to a point in the input cloud and each column corresponds to
      an object ID). Assumes each value is in [0, 1]. Will automatically scale rows to sum to 1.
    - Input completionModel: Trained PCN model for shape completion.
    - Input completedClouds: List of m point clouds, each of shape n_i x 3, estimating the shape of
      each object, corresponding to segmentedClouds: completedClouds[i] =
      completionModel.Predict(segmentedClouds[i]).
    - Input completionCorrectProbabilities: List of m, n_i-element vectors, representing the
      (estimated) probability each point is less than Euclidean distance
      completionModel.errorThreshold from the nearest ground truth point. Values assumed in [0, 1].
    - Input nCompletionSamples: Number of shape completions to sample.
    - Input certainEquivSegmProb: If the maximum segmentation probability (in each row of
      segmentaitonDist) is greater than or equal to certainEquivSegmProb, it is set to 1, and the 
      values in the same row are set to 0.
    - Returns sampleCompletions: List of m lists, each consisting of nCompletionSamples point
      clouds, representing shape samples for each object.
    - Returns sampleNormals: Surface normals corresponding to each cloud in sampleCompletions.
    '''
    
    # Input checking
    
    if len(inputCloud.shape) != 2 or inputCloud.shape[1] != 3:
      raise Exception("inputCloud not 3D.")
      
    if segmentationDist.shape[0] != inputCloud.shape[0]:
      raise Exception(("Mismatch in number of points between segmentationDist (with {} points) " + 
        "and inputCloud (with {} points).").format(segmentationDist.shape[1], inputCloud.shape[1]))
      
    if min(segmentationDist) < 0 or max(segmentationDist) > 1:
      raise Exception("segmentationDist values must be in [0, 1].")
      
    if len(completedClouds) != len(segmentedClouds):
      raise Exception("Mismatch in number of completed clouds and number of segmented clouds.")
      
    if len(completionCorrectProbabilities) != len(segmentedClouds):
      raise Exception("Mismatch in number of objects (completionCorrectProbabilities and " + \
        "segmentedClouds).")
      
    for i in xrange(len(segmentedClouds)):
      if completedClouds[i].shape[0] != completionCorrectProbabilities[i].size:
        raise Exception(("Mismatch in number of points in completedClouds[{}] and " + \
          "completionCorrectProbabilities[{}]).").format(i, i))
          
    if min(completionCorrectProbabilities) < 0 or max(completionCorrectProbabilities) > 1:
      raise Exception("completionCorrectProbabilities not in [0, 1].")
      
    if nCompletionSamples < 0:
      raise Exception("nCompletionSamples must be non-negative.")
      
    if not isinstance(nCompletionSamples, (int, long)):
      raise Exception("nCompletionSamples must be an integer.")
      
    if not isinstance(certainEquivSegmProb, (int, long, float)):
      raise Exception("certainEquivSegmProb must be a scalar.")
    
    # Preprocessing
    
    if self.segmUncertCostFactor > 0:
      
      # assume nearly certainly correct segmentations are certain
      segmentationDist = copy(segmentationDist)
      rowIdx = max(segmentationDist, axis = 1) >= certainEquivSegmProb
      colIdx = argmax(segmentationDist[rowIdx, :], axis = 1)
      certainSegmentaiton = zeros((sum(rowIdx), segmentationDist.shape[1]))
      certainSegmentaiton[arange(certainSegmentaiton.shape[0]), colIdx] = 1.0
      segmentationDist[rowIdx, :] = certainSegmentaiton
      # normalize segmentation probabilities
      segmentationDist /= tile(reshape(sum(segmentationDist, axis = 1),
        (segmentationDist.shape[0], 1)), (1, segmentationDist.shape[1]))
        
      # sample segmentation indices
      colIdx = arange(segmentationDist.shape[1])
      segIdx = array([choice(colIdx, size = nCompletionSamples, p = segmentationDist[i]) for i in \
        xrange(segmentationDist.shape[0])]).T
        
      # find centers of completions
      completedCenters = []
      for cloud in completedClouds:
        completedCenters.append(mean(cloud, axis = 0))
      completedCenters = array(completedCenters)
      
    # initialize output
    sampleCompletions = []; sampleNormals = []
    for i in xrange(len(segmentedClouds)):
      sampleCompletions.append([])
      sampleNormals.append([])
    
    for i in xrange(nCompletionSamples):    
      
      # Sample a segmentation
    
      if self.segmUncertCostFactor > 0:
        
        # extract sampleSegmentation from randomly chosen mask indices
        sampleSegmentation = []
        for j in xrange(segmentationDist.shape[1]):
          obj = inputCloud[segIdx[i] == j, :]
          if obj.shape[0] > 0:
            sampleSegmentation.append(obj)
        
      else:
        # sampling completion only -- use the original segmentation
        sampleSegmentation = copy(segmentedClouds)
      
      # Sample a completion
      
      if self.segmUncertCostFactor > 0:
        
        # perform shape completion for each segment
        sampleCompletion, sampleCompletionProbs = self.CompleteObjects(
          sampleSegmentation, completionModel, False)
          
        # associate sampled completions to original completions, based on nearest centroid
        centers = []
        for cloud in sampleCompletion:
          centers.append(mean(cloud, axis = 0))
        centers = array(centers)
        
        newSampleCompletion = []; newSampleCompletionProbs = []
        for j, completedCenter in enumerate(completedCenters):
          sampleCompletionIdx = argmin(sum(power(centers - tile(reshape(completedCenter, (1, 3)),
            (centers.shape[0], 1)), 2), axis = 1))
          newSampleCompletion.append(sampleCompletion[sampleCompletionIdx])
          newSampleCompletionProbs.append(sampleCompletionProbs[sampleCompletionIdx])
        sampleCompletion = newSampleCompletion; sampleCompletionProbs = newSampleCompletionProbs
        
      else:
        
        # start with original completions, which are already associated with original segmentations
        sampleCompletion = copy(completedClouds)
        sampleCompletionProbs = copy(completionCorrectProbabilities)
      
      if self.compUncertCostFactor > 0:
        
        for j, cloud in enumerate(sampleCompletion):
          
          # Assume each point is offset by a normally distributed signed distance, in a random
          # direction, such that the probability the offset is +/- T is given. We can compute the
          # standard deviation of this distribution by \sigma = T / (erf^{-1}(p) * sqrt(2)).
          # First, we compute the random direction...
          dirOffset = normal(loc = 0.0, scale = 1.0, size = cloud.shape)
          magDirOffset = norm(dirOffset, axis = 1)
          magDirOffset[magDirOffset == 0] = 1.0 # avoid division by 0
          dirOffset /= tile(reshape(magDirOffset, (cloud.shape[0], 1)), (1, 3))
          
          # Next, compute \sigma = T / (erf^{-1}(p) * sqrt(2)), and sample from the resulting
          # normal distribution, and apply the offset.
          sampleCompletionProbs[j][sampleCompletionProbs[j] <= 1.0e-3] = 1.0e-3 # avoid overflow on sigma
          sigma = completionModel.errorThreshold / (sqrt(2) * erfinv(sampleCompletionProbs[j]))
          distOffset = normal(loc = 0.0, scale = sigma)
          cloud += tile(reshape(distOffset, (cloud.shape[0], 1)), (1, 3)) * dirOffset
          
      else:
        # sampling segmentation only -- consider mean shape as certain
        pass
      
      # Compute normals and add to output
      normals = self.env.EstimateNormals(sampleCompletion)
      for j in xrange(len(sampleCompletions)):
        sampleCompletions[j].append(sampleCompletion[j])
        sampleNormals[j].append(normals[j])
      
    return sampleCompletions, sampleNormals
    
  def SaveContinuationData(self, plan, cloud, normals, graspContacts, graspBinormals, cloudSamples,
    normalSamples, triangleNormals, supportingTriangles, cloudCenter, originalGoalPlaces,
    goalPlaces, goalCosts, graspPlaceTable, grasps, graspCosts, tempPlaces, placeCosts,
    obstacleCloudAtStart, obstacleCloudAtGoal):
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
    self.contCloudSamples = cloudSamples
    self.contNormalSamples = normalSamples
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