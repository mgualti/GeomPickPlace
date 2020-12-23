'''TODO'''

# python
from time import time
from multiprocessing.connection import Client
# scipy
from numpy.linalg import norm
from numpy.random import normal, shuffle, uniform
from numpy import arange, array, argsort, dot, vstack
# openrave
import openravepy
# self
from planner import Planner
import point_cloud

class PlannerPacking(Planner):

  def __init__(self, env, nGraspSamples, loweringDelta, collisionCubeSize, nWorkers,
    maxGoalsPerWorker, maxPlanningTime, minPointsPerSegment, preGraspOffset):
    '''TODO'''
    
    Planner.__init__(self, env, minPointsPerSegment, preGraspOffset)
    
    # internal variables
    self.obstacleCloud = None
    
    # parameters (grasping)
    self.nGraspSamples = nGraspSamples
    
    # parameters (planning)
    self.loweringDelta = loweringDelta
    self.collisionCubeSize = collisionCubeSize
    self.maxPlanningTime = maxPlanningTime
    self.nWorkers = nWorkers
    self.maxGoalsPerWorker = maxGoalsPerWorker
    self.boxExtents = env.boxExtents
    self.boxPosition = env.boxPosition
    
  def AddObstacleCloud(self, cloud):
    '''TODO'''
    
    self.RemoveObstacleCloud()
    self.env.AddObstacleCloud(cloud, self.collisionCubeSize)
    self.obstacleCloud = self.env.obstacleCloud
    self.env.obstacleCloud = None
    self.obstacleCloud.SetName("PlannerPackingObstacleCloud")
    # assumes no ground truth object models are in the bin!
    
  def GetGoalPoses(self, segmClouds, segmCorrectProbs, compClouds, compCorrectProbs, normals,
    obstacleCloud, regraspPlanner):
    '''TODO'''
    
    # 1. Start clock and error checking.
    
    startTime = time()
    
    if len(segmClouds) == 0:
      print("No objects were detected!")
      return [], [], []
      
    if len(segmClouds) != len(compClouds) or len(segmClouds) != len(segmCorrectProbs) or \
      len(segmClouds) != len(compCorrectProbs):
      raise Exception("Inconsistent lenghths in inputs.")
    
    # 2. Find grasps on all objects, for later reachability checking.
    
    pointCosts = regraspPlanner.GetPointCosts(
      segmClouds, compClouds, segmCorrectProbs, compCorrectProbs)    
    
    nGrasps = 0
    while nGrasps == 0 and time() - startTime < self.maxPlanningTime:
      print("Sampling {} grasps on all objects.".format(self.nGraspSamples))
      grasps = regraspPlanner.PreSampleGraspsOnObjects(
        compClouds, normals, pointCosts, self.nGraspSamples)
      for graspList in grasps: nGrasps += len(graspList)
    
    if nGrasps == 0:
      print("No graps found!")
      return [], [], []
    
    # 3. Call a worker for each solution.
    
    # pass input data to workers
    connections = []
    for i in xrange(self.nWorkers):
      
      message = ["packing", startTime, compClouds, obstacleCloud, grasps]
        
      try:
        print("Starting pack planning worker {}.".format(i))
        connection = Client(('localhost', 7000 + i))
      except:
        for connection in connections: connection.close()
        raise Exception("Unable to connect to worker {}.".format(i))
      connections.append(connection)
      connections[-1].send(message)
      
    # wait for result
    goalPoses = []; goalCosts = []; targObjIdxs = []
    for i, connection in enumerate(connections):
      
      # unpack message
      message = connection.recv()
      connection.close()
      purpose = message[0]
      data = message[1:]
      
      # save solution
      if purpose == "packing":
        goalPoses += data[0]
        goalCosts += data[1]
        targObjIdxs += data[2]
        print("Worker {} completed with heights (cm) {}.".format(i, array(data[1]) * 100))
      else:
        print("Warning: received purpose {} when expecting purpose packing.".format(purpose))
    
    # 4. Return solutions.
    
    return goalPoses, goalCosts, targObjIdxs
    
  def GetGoalPosesWorker(self, startTime, clouds, inputObstacleCloud, grasps):
    '''Random search for minimum-height packing.'''
    
    # while no solution or time remains:
    goalPoses = []; goalCosts = []; targObjIdxs = []
    while time() - startTime < self.maxPlanningTime:
    
      # randomize object ordering
      idx = arange(len(clouds))
      shuffle(idx)
      
      # add the input cloud as an obstacle and compute its height
      obstacleCloud = inputObstacleCloud
      self.AddObstacleCloud(obstacleCloud)
      inputObstacleHeight = -float('inf') if inputObstacleCloud.shape[0] == 0 else \
        max(inputObstacleCloud[:, 2]) - self.env.GetTableHeight() - self.boxExtents[3]
      
      # for each object:
      thisSolution = None
      for j, i in enumerate(idx):
        
        # do not consider objects with no grasps
        if len(grasps[i]) == 0: continue
        
        # while object in collision or has no reachable grasps:        
        while True:
          
          # exit if the time limit has been reached
          if time() - startTime >= self.maxPlanningTime:
            self.RemoveObstacleCloud()
            return goalPoses, goalCosts, targObjIdxs
          
          # determine object pose above the box
          T = self.SampleObjectOrientation()
          movedCloud = point_cloud.Transform(T, clouds[i])
          p = self.SampleObjectPosition(movedCloud)
          T[0:3, 3] = p
          
          # lower object until it makes contact
          movedCloud = point_cloud.Transform(T, clouds[i])
          z = self.LowerObject(movedCloud)
          T[2, 3] += z
          
          # check reachability (first object only)
          movedCloud = point_cloud.Transform(T, clouds[i])
          if self.IsReachable(T, grasps[i]):
            # this object is OK
            if thisSolution is None: thisSolution = (T, i)
            obstacleCloud = vstack([obstacleCloud, movedCloud])
            self.AddObstacleCloud(obstacleCloud)
            break
        
        # abandon this attempt if there are enough solutions and this is worse than worst
        thisCost = max(obstacleCloud[:, 2]) - self.env.GetTableHeight() - self.boxExtents[3]
        if len(goalCosts) >= self.maxGoalsPerWorker and thisCost >= goalCosts[-1]: break
        
      # if more solutions are needed or this height is better than the worst current height, save it
      if len(goalCosts) < self.maxGoalsPerWorker or thisCost < goalCosts[-1]:
        # save this solution
        goalPoses.append(thisSolution[0])
        goalCosts.append(thisCost)
        targObjIdxs.append(thisSolution[1])
        # sort current solutions by lowest cost
        idx = argsort(goalCosts)
        goalPoses = [goalPoses[i] for i in idx]
        goalCosts = [goalCosts[i] for i in idx]
        targObjIdxs = [targObjIdxs[i] for i in idx]
        # if there are too many solutions, remove the worst one
        if len(goalPoses) > self.maxGoalsPerWorker:
          goalPoses = goalPoses[0:-1]
          goalCosts = goalCosts[0:-1]
          targObjIdxs = targObjIdxs[0:-1]
        # print current costs
        print("Heights improved to (cm): {}.".format(array(goalCosts) * 100))
      
      # clean up this attempt
      self.RemoveObstacleCloud()
      
      # exit early if:
      #   (a) there are enough solutions and
      #   (b) the greatest height is lower than the height of the current obstacle
      if len(goalCosts) >= self.maxGoalsPerWorker and goalCosts[-1] <= inputObstacleHeight:
        print("For each solution, all objects can be placed below current tallest object.")
        break
      
    return goalPoses, goalCosts, targObjIdxs
    
  def IsReachable(self, T, grasps):
    '''TODO'''
    
    movedGrasps = []
    for grasp in grasps:
      movedGrasps.append(dot(T, grasp))
    movedPreGrasps = self.GetPreGrasps(movedGrasps)
    reachable = self.env.AreAnyReachable(movedGrasps, movedPreGrasps, ignoreObjects = True)
    self.env.MoveRobotToHome()
    return reachable
    
  def LowerObject(self, cloud):
    '''TODO'''
    
    # prepare to lower object
    body = self.env.AddObstacleCloud(cloud, self.collisionCubeSize)
    Toriginal = body.GetTransform()
    Tnew = body.GetTransform()
    z = 0.0
    
    # lower (or lift) object
    collision = self.env.env.CheckCollision(body)
    
    if collision:
      # move up until not in collision
      while True:
        z += self.loweringDelta
        Tnew[2, 3] = Toriginal[2, 3] + z
        body.SetTransform(Tnew)
        if not self.env.env.CheckCollision(body):
          break
    
    else:
      # move down until in collision (then move up 1)
      while True:
        z -= self.loweringDelta
        Tnew[2, 3] = Toriginal[2, 3] + z
        body.SetTransform(Tnew)
        if self.env.env.CheckCollision(body):
          break
      z += self.loweringDelta
    
    # cleanup
    self.env.RemoveObstacleCloud()
    return z
    
  def RemoveObstacleCloud(self):
    '''TODO'''
    
    if self.obstacleCloud is not None:
      with self.env.env:
        self.env.env.Remove(self.obstacleCloud)
      self.obstacleCloud = None
    
  def SampleObjectOrientation(self):
    '''Sample object orientation uniformly. To achieve this, sample a point from a 4D standard
       normal distribution, then normalize.
    - Returns T: Homogeneous transform with 0 position and orientation sampled uniformly at random.
    '''
    
    q = normal(size = 4)
    q /= norm(q)    
    
    return openravepy.matrixFromQuat(q)
    
  def SampleObjectPosition(self, cloud):
    '''TODO'''
    
    extents = [(min(cloud[:, i]), max(cloud[:, i])) for i in xrange(3)]
    
    xl = self.boxPosition[0] - self.boxExtents[0] / 2.0 - extents[0][0] + self.collisionCubeSize
    xh = self.boxPosition[0] + self.boxExtents[0] / 2.0 - extents[0][1] - self.collisionCubeSize
    
    yl = self.boxPosition[1] - self.boxExtents[1] / 2.0 - extents[1][0] + self.collisionCubeSize
    yh = self.boxPosition[1] + self.boxExtents[1] / 2.0 - extents[1][1] - self.collisionCubeSize
    
    px = uniform(xl, xh)
    py = uniform(yl, yh)
    pz = self.boxPosition[2] + self.boxExtents[2] / 2.0 - extents[2][0] + self.collisionCubeSize
    
    return [px, py, pz]