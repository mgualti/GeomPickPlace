'''TODO'''

# python
from time import time
from multiprocessing.connection import Client
# scipy
from numpy import array
# self
from planner_packing import PlannerPacking

class PlannerPackingMC(PlannerPacking):

  def __init__(self, env, nGraspSamples, loweringDelta, collisionCubeSize, nWorkers,
    maxGoalsPerWorker, maxPlanningTime, minPointsPerSegment, preGraspOffset):
    '''TODO'''
    
    PlannerPacking.__init__(self, env, nGraspSamples, loweringDelta, collisionCubeSize, nWorkers,
      maxGoalsPerWorker, maxPlanningTime, minPointsPerSegment, preGraspOffset)
    
  def GetGoalPoses(self, clouds, normals, completionSamples, completionNormals, obstacleCloud,
    regraspPlanner):
    '''TODO'''
    
    # 1. Start clock and error checking.
    
    startTime = time()
    
    if len(clouds) == 0:
      print("No objects were detected!")
      return [], [], []
      
    if len(clouds) != len(normals) or len(clouds) != len(completionSamples) or \
      len(clouds) != len(completionNormals):
      raise Exception("Inconsistent lenghths in inputs.")
    
    # 2. Find grasps on all objects, for later reachability checking. 
    
    nGrasps = 0
    while nGrasps == 0 and time() - startTime < self.maxPlanningTime:
      print("Sampling {} grasps on all objects.".format(self.nGraspSamples))
      grasps = regraspPlanner.PreSampleGraspsOnObjects(
        clouds, normals, completionSamples, completionNormals, self.nGraspSamples)
      for graspList in grasps: nGrasps += len(graspList)
    
    if nGrasps == 0:
      print("No graps found!")
      return [], [], []
    
    # 3. Call a worker for each solution.
    
    # pass input data to workers
    connections = []
    for i in xrange(self.nWorkers):
      
      message = ["packing", startTime, clouds, obstacleCloud, grasps]
        
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