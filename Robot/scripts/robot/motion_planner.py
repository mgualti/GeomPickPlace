'''This module is a generic motion planner that uses trajopt.'''

# python
# scipy
import numpy # scipy
from numpy.linalg import norm
from numpy import arccos, array, cross, dot, eye, linspace, ones, sqrt
# ros
import tf
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
# trajopt
import json
import trajoptpy
import openravepy
# self
import plot_rviz

class MotionPlanner:
  '''View planner base class.'''
  
  def __init__(self, maxCSpaceJump, timeout):
    '''Constructor for view planner.'''
    
    self.maxCSpaceJump = maxCSpaceJump
    self.timeout = timeout
    self.trajectoryPub = rospy.Publisher("/trajectory", Marker, queue_size=1)
    self.samplesPub = rospy.Publisher("/planning_samples", Marker, queue_size=1)
  
  def DrawTrajectory(self, traj, env, rgb, indices=[-1,-1]):
    '''Draw a given trajectory in rviz.'''
    pts = []
    for q in traj:
      pts.append(env.CalcFk(q)[0:4, 3])
    
    marker = plot_rviz.CreateMarker("trajectory", 0, "base_link", rospy.Duration(60), [0, 0, 1], 1)
    
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.02
    marker.scale.y = 0.02
  
    for i in xrange(len(pts)):
      marker.points.append(Point(pts[i][0],pts[i][1],pts[i][2]))
    
    self.trajectoryPub.publish(marker)

  def HierarchicalPlan(self, startConfig, targetConfig, env):
    '''Run the hierarchical planner.'''
    
    # basic checking
    if not env.IsInJointLimits(startConfig):
      print("HierarchicalPlan: The start config is outside the joint limits.")
      return False, []
    if env.IsInCollision(startConfig):
      print("HierarchicalPlan: The start config is in collision.")
      return False, []
    if not env.IsInJointLimits(targetConfig):
      print("HierarchicalPlan: The target config is outside the joint limits.")
      return False, []
    if env.IsInCollision(targetConfig):
      print("HierarchicalPlan: The target config is in collision.")
      return False, []
    
    # use linear planner
    traj = [startConfig, targetConfig]
    safe, info = IsPathSafe(traj, env, getAllInfo=False)
    method_type = "linear"
    
    # switch to trajopt
    if not safe:
      traj, safe = self.PlanTrajopt(startConfig, targetConfig, env)
      method_type = "Trajopt"
    
    # switch to RRT
    if not safe:
      traj, safe = self.PlanRrt(startConfig, targetConfig, env)
      method_type = "RRT"
    
    if not safe:
      print("Hierarchical planner could not generate a valid trajectory.")
      return False, []
    
    # visualize trajectory
    self.DrawTrajectory(traj, env, [0.0, 0.0, 0.5])
    print("Using {} planner. Trajectory length: {}".format(method_type, len(traj)))
    
    return True, traj
    
  def PlanRrt(self, configStart, configEnd, mover):
    '''Uses OMPL's RRT* to find a motion plan.'''
    
    planner = openravepy.RaveCreatePlanner(mover.env, 'OMPL_RRTstar')

    # Setup the planning instance.
    params = openravepy.Planner.PlannerParameters()

    params.SetRobotActiveJoints(mover.robot)
    params.SetInitialConfig(configStart)
    params.SetGoalConfig(configEnd)

    # Set the timeout and planner-specific parameters. You can view a list of
    # supported parameters by calling: planner.SendCommand('GetParameters')
    params.SetExtraParameters('<range>' + str(self.maxCSpaceJump) + '</range>')
    params.SetExtraParameters('<time_limit>' + str(self.timeout)+ '</time_limit>')
    planner.InitPlan(mover.robot, params)
    
    # Invoke the planner.
    traj = openravepy.RaveCreateTrajectory(mover.env, '')
    result = planner.PlanPath(traj)
    if not result == openravepy.PlannerStatus.HasSolution:
      print("No RRT solution found.")
      return [], False
    trajectory = []
    for i in xrange(traj.GetNumWaypoints()):
        trajectory.append(traj.GetWaypoint(i)[:6])
    
    # check collision
    safe, info = IsPathSafe(trajectory, mover)
    if not safe:
      print("RRT collision info: ")
      print(info)
      return trajectory, safe
    
    # reduce trajectory
    trajectory = ReduceTrajectory(trajectory, mover)    
    return trajectory, safe
  
  def PlanTrajopt(self, configStart, configEnd, env):
    '''Plan a trajectory between two given arm configurations.'''
    
    trajLen = int((1.0 / self.maxCSpaceJump) * norm(configStart - configEnd)) + 1
    
    request = {
      "basic_info" :
        {"n_steps" : trajLen, "robot": "ur5", "manip" : "arm", "start_fixed" : True},
      "costs" : [
        {"type" : "joint_vel", "params" : {"coeffs" : [1]}},
        {"type" : "collision", "name" : "cont_coll", "params" :
          {"continuous" : False, "coeffs" : [1], "dist_pen" : [0.02]}}
        ],
      "constraints" : [
      {"type" : "joint", "params" : {"vals" : configEnd.tolist() }
      }
      ],
      "init_info" : {"type" : "straight_line", "endpoint" : configEnd.tolist()}
    }
    
    env.robot.SetDOFValues(configStart, env.manip.GetArmIndices())
    s = json.dumps(request) # convert dictionary into json-formatted string
    prob = trajoptpy.ConstructProblem(s, env.env)
    result = trajoptpy.OptimizeProblem(prob)
    trajectory = result.GetTraj()
    
    # check collision
    safe, info = IsPathSafe(trajectory, env)
    if not safe:
      print("Trajopt collision info: ")
      print(info)
      return trajectory, safe
    
    # reduce trajectory
    trajectory = ReduceTrajectory(trajectory, env)
    return trajectory, safe

# UTILITIES=============================================================================================================

def ClosestDistance(configs1, configs2):
  '''Given two lists of configurations, finds the distance between the closest pair.
  
  - Input configs1: List of configurations.
  - Input configs2: List of configurations.
  - Returns dBest: Shortest L2 distrance between configs1 and configs2.
  '''
  
  dBest = float('inf')
  for c1 in configs1:
    for c2 in configs2:
      v = c1-c2
      d = dot(v,v)
      if d < dBest: dBest = d
  return sqrt(dBest)

def ClosestPair(configs1, configs2):
  '''Finds a pair of configurations that are closest.
  
  - Input configs1: List of candidates for the first item in the pair. (List of numpy arrays.)
  - Input configs2 List of candidates for the second item in the pair. (List of numpy arrays.)
  - Returns closestPair: Two-element tuple of the closest configurations (L2 in c-space).
  '''
  
  dBest = float('inf'); c1Best = None; c2Best = None
  for c1 in configs1:
    for c2 in configs2:
      v = c1-c2
      d = dot(v,v)
      if d < dBest: c1Best = c1; c2Best = c2; dBest = d
  
  return (c1Best, c2Best)

def ClosestPairIndices(configs1, configs2):
  '''Finds a pair of configurations that are closest.
  
  - Input configs1: List of candidates for the first item in the pair. (List of numpy arrays.)
  - Input configs2 List of candidates for the second item in the pair. (List of numpy arrays.)
  - Returns closestPair: Two-element tuple of the closest configurations (L2 in c-space).
  '''
  
  dBest = float('inf'); c1Best = -1; c2Best = -1
  for i in xrange(len(configs1)):
    for j in xrange(len(configs2)):
      v = configs1[i] - configs2[j]
      d = dot(v,v)
      if d < dBest: c1Best = i; c2Best = j; dBest = d
  
  return (c1Best, c2Best)
  
def ComputePathCost(trajectory):
  '''Computes sum of L2 distances between points in a trajectory.
  
  - Input trajectory: List of joint configurations (numpy arrays).
  - Returns pathCost: Sum of L2 distances between points in the trajectory.
  '''
  
  pathCost = 0; config0 = trajectory[0]
  for config1 in trajectory:
    pathCost += norm(config0-config1)
    config0 = config1
  return pathCost

def IsInWorkspaceEllipsoid(A,c,p,tol=0.001):
  '''Determines if a point lies inside of an ellipsoid
  - Input A: 3x3 matrix describing the (not necessarily axis-alinged) ellipsoid.
  - Input c: 3-element vector describing the origin of the ellipsoid.
  - Input p: A point in 3d to check for membership.
  - Input tol: How much error is tolerated in membership check (units?)
  - Returns: True if the point lies within or on the ellipsoid.
  - See: http://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
  - See: https://en.wikipedia.org/wiki/Ellipsoid
  '''
  
  v = p-c
  return dot(dot(v,A),v) <= 1+tol

def GeneratePose(sensorPosition, targetPosition, l=(1,0,0)):
  '''Helps to determine a sensor pose just given a sensor position and view target.
    
  - Input sensorPosition: 3-element desired position of sensor placement.
  - Input targetPosition: 3-element position of object required to view.
  - Input l: Sensor LOS axis in base frame given identity orientation.
  - Returns T: 4x4 numpy array (transformation matrix) representing desired pose of end effector in the base frame.
  '''
  
  v = targetPosition - sensorPosition
  
  vMag = norm(v)
  v = [v[0]/vMag, v[1]/vMag, v[2]/vMag]
  
  k = [l[1]*v[2]-l[2]*v[1], l[2]*v[0]-l[0]*v[2], l[0]*v[1]-l[1]*v[0]]
  theta = arccos(l[0]*v[0] + l[1]*v[1] + l[2]*v[2])
  
  q = tf.transformations.quaternion_about_axis(theta, k)
  return openravepy.matrixFromPose(numpy.r_[[q[3],q[0],q[1],q[2]], sensorPosition])
  
def GeneratePoseGivenUp(sensorPosition, targetPosition, upAxis):
  '''Generates the sensor pose with the LOS pointing to a target position and the "up" close to a specified up.
    
  - Input sensorPosition: 3-element desired position of sensor placement.
  - Input targetPosition: 3-element position of object required to view.
  - Input upAxis: The direction the sensor up should be close to.
  - Returns T: 4x4 numpy array (transformation matrix) representing desired pose of end effector in the base frame.
  '''
  
  v = targetPosition - sensorPosition
  v = v / norm(v)
  
  u = upAxis - dot(upAxis, v) * v
  u = u / norm(u)
  
  t = cross(u, v)
  
  T = eye(4)
  T[0:3,0] = v
  T[0:3,1] = t
  T[0:3,2] = u
  T[0:3,3] = sensorPosition
  
  return T
  
def IsPathSafe(trajectory, env, getAllInfo=True, step=0.02):
  '''Checks if the trajectory is both collision-free and within the joint limits.'''
  
  safe = True; unsafeInfo = []
  
  if len(trajectory) == 0:
    return safe, unsafeInfo
  
  if len(trajectory) == 1:
    if env.IsInCollision(trajectory[0]):
      safe = False
      unsafeInfo.append( ((0,0), "collision") )
      if not getAllInfo: return safe, unsafeInfo
    if not env.IsInJointLimits(trajectory[0]):
      safe = False
      unsafeInfo.append( ((0,0), "joints") )
      if not getAllInfo: return safe, unsafeInfo
  
  for i in xrange(1, len(trajectory)):
    
    dist = norm(trajectory[i-1] - trajectory[i])
    nStep = int(round(dist / step))
    alpha = linspace(0, 1, nStep)

    for j in xrange(nStep):
      config = alpha[j]*trajectory[i] + (1.0-alpha[j])*trajectory[i-1]
      if env.IsInCollision(config):
        safe = False
        unsafeInfo.append( ((i-1, j), "collision") )
        if not getAllInfo: return safe, unsafeInfo
      if not env.IsInJointLimits(config):
        safe = False
        unsafeInfo.append( ((i-1, j), "joints") )
        if not getAllInfo: return safe, unsafeInfo
  
  return safe, unsafeInfo

def IsTravelOnLine(trajectory, env, maxDistFromLine, maxAngleFromLine, step=0.02):
  '''Simulates linear controller and checks if the end effector travels in a straight line between waypoints.'''
  
  if len(trajectory) < 2:
    return True
  
  for i in xrange(1, len(trajectory)):

    # first point in Euclidean space
    T0 = env.CalcFk(trajectory[i-1])
    p0 = T0[0:3, 3]
    q0 = openravepy.quatFromRotationMatrix(T0)
    
    # second point in Euclidean space
    T1 = env.CalcFk(trajectory[i])
    p1 = T1[0:3, 3]
    q1 = openravepy.quatFromRotationMatrix(T1)
    
    # compute evenly spaced waypoints given step size
    dist = norm(trajectory[i-1] - trajectory[i])
    nStep = int(round(dist / step))
    alpha = linspace(0, 1, nStep)    
    
    for j in xrange(nStep):
      
      # intermediate configuration
      config = alpha[j]*trajectory[i] + (1.0-alpha[j])*trajectory[i-1]    
    
      # middle point Euclidean space
      T = env.CalcFk(config)
      p = T[0:3,3]
      q = openravepy.quatFromRotationMatrix(T1)
      
      # compute distances
      distPos = norm(cross(p1-p0, p0-p)) / norm(p1-p0)
      distOrient = max(norm(q - q0), norm(q - q1))
      
      if distPos > maxDistFromLine:
        print("Not on line -- distance: {}".format(distPos))
        return False
      if distOrient > maxAngleFromLine:
        print("Not on line -- angle: {}".format(distOrient))
        return False
  
  return True

def RayIsClear(camPoint, targPoint, obstacleTree, raySegLen=0.015, reliefFromTarget=0.04):
  '''Returns True if a ray from a camera to a target point is clear in a point cloud.
  
  - Input camPoint: 3-element vector of the array origin.
  - Input targPoint: 3-element vector of the array destination.
  - Input obstacleTree: KDTree object containing the point cloud representing an opaque obstacle.
  - Input raySegLen: Determines how big the ray tube is that must be clear and also the step size.
  - Returns True if the ray is clear and False if a point is in the tube between points.
  '''
  
  ray = camPoint - targPoint
  rayMag = norm(ray)
  unitRay = ray / rayMag
  
  targPoint = targPoint + reliefFromTarget*unitRay
  
  currentLength = raySegLen
  endLength = rayMag - raySegLen
  
  while currentLength < endLength:
    queryPoint = targPoint + unitRay*currentLength
    d = obstacleTree.query(queryPoint)
    # exit early if not visible
    if d[0] < raySegLen: return False
    # exit early if closest point is further than line length
    if d[0] > endLength-currentLength: return True
    currentLength += raySegLen
  
  return True

def ReduceTrajectory(trajectory, mover):
  '''Minimizes the path length as much as possible while avoiding collisions.'''
  
  rootIdx = 0
  deleteIndices = []
  for i in xrange(1, len(trajectory)-1):
    
    safe, unsafeInfo = IsPathSafe([trajectory[rootIdx], trajectory[i+1]], mover, getAllInfo=False)
    
    if safe: # this point is unnecessary
      deleteIndices.append(i)
    else: # this point is necessary
      rootIdx = i
  
  reducedTrajectory = []
  for i, point in enumerate(trajectory):
    if i not in deleteIndices:
      reducedTrajectory.append(point)
    
  return reducedTrajectory