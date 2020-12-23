'''A class for regrasp planning. Extends ideas in "Regrasping" by Tournassoud et al., 1987.'''

# python
# scipy
import matplotlib
from numpy import exp, isnan, log, sum, zeros
# openrave
# self
import hand_descriptor
from pcn_model_sp import PcnModelSP
from planner_regrasp import PlannerRegrasp
  
class PlannerRegraspSP(PlannerRegrasp):

  def __init__(self, env, temporaryPlacePosition, nGraspSamples, halfAngleOfGraspFrictionCone,
    nTempPlaceSamples, nGoalPlaceSamples, modelFileNameGraspPrediction,
    modelFileNamePlacePrediction, taskCostFactor, stepCost, graspUncertCostFactor,
    placeUncertCostFactor, insignificantCost, maxSearchDepth, maxIterations, minPointsPerSegment,
    preGraspOffset):
    '''TODO'''
    
    PlannerRegrasp.__init__(self, env, temporaryPlacePosition, nGraspSamples,
      halfAngleOfGraspFrictionCone, nTempPlaceSamples, nGoalPlaceSamples, taskCostFactor, stepCost,
      0, 0,  graspUncertCostFactor, placeUncertCostFactor, 0, insignificantCost, maxSearchDepth,
      maxIterations, minPointsPerSegment, preGraspOffset)
    
    # input checking
    if not isinstance(modelFileNameGraspPrediction, str):
       raise Exception("Expected str modelFileNameGraspPrediction; got {}.".format(\
         type(modelFileNameGraspPrediction)))
         
    if not isinstance(modelFileNamePlacePrediction, str):
       raise Exception("Expected str modelFileNamePlacePrediction; got {}.".format(\
         type(modelFileNameGraspPrediction)))
    
    # load grasp and place prediction models
    self.modelGraspPrediction = PcnModelSP(128, -1, modelFileNameGraspPrediction)
    self.modelPlacePrediction = PcnModelSP(1024, -1, modelFileNamePlacePrediction)
    
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
      
      graspsNew, _ = self.SampleGrasps(cloud, normals, graspContacts, graspBinormals)
      graspCostsNew = self.GetGraspCosts(cloud, graspsNew)
        
      if nReachableStart > 0 and nReachableGoal > 0:
        tempPlacesNew, _ = self.SamplePlaces(cloud, supportingTriangles, triangleNormals, cloudCenter)
        tempPlaceCostsNew = self.GetPlaceCosts(cloud, tempPlacesNew)
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
    
  def GetCostLowerBound(self, cloud, normals, pointCosts, pointPairs, goalPlaceCosts):
    '''TODO'''
    
    return 2.0 * self.stepCost + self.insignificantCost
    
  def GetGraspCosts(self, cloud, grasps):
    '''TODO'''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("Expected 3D cloud")
      
    if not isinstance(grasps, list):
      raise Exception("Expected grasps to be of type list; got type {}.".format(type(grasps)))
    
    # get grasp costs
    graspProbs = self.modelGraspPrediction.PredictGrasps(cloud, grasps)
    graspCosts = -log(graspProbs)
    return 0.5 * self.graspCostFactor * graspCosts
    
  def GetPlaceCosts(self, cloud, places):
    '''TODO'''
    
    # input checking
    if len(cloud.shape) != 2 or cloud.shape[1] != 3:
      raise Exception("Expected 3D cloud")
      
    if not isinstance(places, list):
      raise Exception("Expected places to be of type list; got type {}.".format(type(places)))
    
    # get place costs
    placeProbs = self.modelPlacePrediction.PredictPlaces(cloud, places, self.temporaryPlacePosition)
    placeCosts = -log(placeProbs)
    return 0.5 * self.placeCostFactor * placeCosts
    
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
        graspsNew, _ = self.SampleGrasps(cloud, normals, graspContacts, graspBinormals)
        graspCostsNew = self.GetGraspCosts(cloud, graspsNew)
      else:
        notFirstLoop = True
        
      if nReachableStart > 0 and nReachableGoal > 0:
        tempPlacesNew, _ = self.SamplePlaces(cloud, supportingTriangles, triangleNormals, cloudCenter)
        tempPlaceCostsNew = self.GetPlaceCosts(cloud, tempPlacesNew)
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
      grasps, _ = self.SampleGrasps(clouds[i], normals[i], pairs, binormals)
      self.nGraspSamples = tmpSamples
      graspCosts = self.GetGraspCosts(clouds[i], grasps)
      
      self.preSampledPairs.append(pairs)
      self.preSampledBinormals.append(binormals)
      self.preSampledGrasps.append(grasps)
      self.preSampledGraspCosts.append(graspCosts)
      
      if visualize:
        self.env.PlotDescriptors(hand_descriptor.DescriptorsFromPoses(grasps),
          matplotlib.cm.viridis(exp(-graspCosts)))
        raw_input("Grasps for object {}.".format(i))
        
    return self.preSampledGrasps