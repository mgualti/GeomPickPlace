'''TODO'''

# python
import os
from copy import copy
# scipy
from numpy.linalg import inv, norm
from numpy import argmin, array, ascontiguousarray, cos, cross, dot, empty, eye, logical_and, \
  pi, power, sum, sin, tile, vstack, zeros
# openrave
import openravepy
# self
import point_cloud
import c_extensions
from hand_descriptor import HandDescriptor

class Environment(object):

  def __init__(self, showViewer, showWarnings):
    '''Initializes OpenRAVE environment.
    - Input showViewer: If True, shows the QT viewer.
    '''

    # parameters
    self.showViewer = showViewer
    projectDir = os.getcwd() + "/"

    # create openrave environment
    self.env = openravepy.Environment()
    if not showWarnings: self.env.SetDebugLevel(0)
    if showViewer: self.env.SetViewer('qtcoin')
    self.env.Load(projectDir + "openrave/environment.xml")
    self.robot = self.env.GetRobots()[0]
    self.manip = self.robot.GetActiveManipulator()
    self.gTe = dot(point_cloud.InverseTransform(self.robot.GetLink("gripper_center").GetTransform()),
      self.robot.GetLink("ee_link").GetTransform())
    self.homeConfig = self.robot.GetDOFValues()[0:6]
    self.hasRobot = True
    
    # add floating hand
    self.env.Load(projectDir + "openrave/floating_hand.xml")
    self.floatingHand = self.env.GetRobots()[2]
    self.gcTfh = dot(point_cloud.InverseTransform(
      self.floatingHand.GetLink("gripper_center").GetTransform()), self.floatingHand.GetTransform())
    self.hasFloatingHand = True
      
    # joint values and joint limits
    self.jointLimits = zeros((2, 7))
    self.jointLimits[0][0] = -pi;     self.jointLimits[1][0] = pi
    self.jointLimits[0][1] = -3*pi/2; self.jointLimits[1][1] = pi/2
    self.jointLimits[0][2] = -2.8;    self.jointLimits[1][2] = 2.8
    self.jointLimits[0][3] = -4.15;   self.jointLimits[1][3] = 1.0
    self.jointLimits[0][4] = -2.18;   self.jointLimits[1][4] = 2.18
    self.jointLimits[0][5] = -pi;     self.jointLimits[1][5] = pi
    self.jointLimits[0][6] =  0.00;   self.jointLimits[1][5] = 0.80
    self.robot.SetDOFLimits(self.jointLimits[0], self.jointLimits[1])
    self.jointLimits = self.jointLimits[:, 0:6]

    # set collision checker options
    collisionChecker = openravepy.RaveCreateCollisionChecker(self.env, 'ode')
    self.env.SetCollisionChecker(collisionChecker)

    # set physics options
    #self.physicsEngine = openravepy.RaveCreatePhysicsEngine(self.env, "ode")
    #self.env.SetPhysicsEngine(self.physicsEngine)
    #self.env.GetPhysicsEngine().SetGravity([0.0, 0.0, -9.8])
    self.env.StopSimulation()
    
    # get sensor from openrave
    self.sensorRobot = self.env.GetRobots()[1]
    self.sensor = self.env.GetSensors()[0]
    self.hasSensor = True
    for link in self.sensorRobot.GetLinks():
      link.SetStatic(True)

    # table(s)
    self.table = self.env.GetKinBody("table")
    self.tablePosition = self.table.GetTransform()[0:3, 3]
    self.tableExtents = self.table.ComputeAABB().extents()
    self.tableHeight = self.tablePosition[2] + self.tableExtents[2]
    self.hasTable = True

    # Internal state
    self.obstacleCloud = None
    self.attachedObject = None
    self.plotCloudHandle = None
    self.plotDescriptorsHandle = None
    
  def AttachObject(self, cloud, config, cubeSize):
    '''TODO'''
    
    if cloud.shape[0] == 0:
      body = openravepy.RaveCreateKinBody(self.env, "")
    else:
      cloud = point_cloud.Voxelize(cubeSize, cloud)
      boxInfo = empty((cloud.shape[0], 6))
      boxInfo[:, 0:3] = cloud
      boxInfo[:, 3:6] = cubeSize / 2.0
      body = openravepy.RaveCreateKinBody(self.env, "")
      body.InitFromBoxes(boxInfo, True)
    
    body.SetName("AttachedObject")
    self.env.Add(body)
    self.attachedObject = body
    self.MoveRobot(config)
    self.robot.Grab(body)
    return
    
  def AddFloatingHand(self):
    '''TODO'''
    
    if self.hasFloatingHand: return
    self.env.Add(self.floatingHand)
    self.hasFloatingHand = True
    
  def AddRobot(self):
    '''TODO'''
    
    if self.hasRobot: return
    self.env.Add(self.robot)
    self.hasRobot = True
    
  def AddSensor(self):
    '''TODO'''
    
    if self.hasSensor: return
    self.env.Add(self.sensorRobot)
    self.hasSensor = True
    
  def AddTable(self):
    
    if self.hasTable: return
    self.env.Add(self.table)
    self.hasTable = True
    
  def AddObstacleCloud(self, cloud, cubeSize):
    '''Downsamples a point cloud and adds it as an obstacle.
    - Input cloud: nx3 numpy array to add as an obstacle.
    - Input cubeSize: Size of the cubes which will be centered on the downsampled points.
    - Returns body: KinBody object representing the obstacle.
    '''
    
    self.RemoveObstacleCloud()
    
    if cloud.shape[0] == 0:
      body = openravepy.RaveCreateKinBody(self.env, "")
    else:
      cloud = point_cloud.Voxelize(cubeSize, cloud)
      boxInfo = empty((cloud.shape[0], 6))
      boxInfo[:, 0:3] = cloud
      boxInfo[:, 3:6] = cubeSize / 2.0
      body = openravepy.RaveCreateKinBody(self.env, "")
      body.InitFromBoxes(boxInfo, True)
    
    body.SetName("ObstacleCloud")
    self.obstacleCloud = body
    self.env.Add(body)
    return body
    
  def CalcFk(self, config, endEffectorName = None):
    '''Calculate the Cartesian pose for a given collection of joint positions.'''
    
    if endEffectorName is None:
      endEffectorName = "ee_link"
    
    self.robot.SetDOFValues(config, self.manip.GetArmIndices())
    return self.robot.GetLink(endEffectorName).GetTransform()
    
  def CalcIkForT(self, T):
    '''Computes inverse kinematiccs for UR5 arm, accounting for joint limits.
    - Input T: Desired pose of gripper_center in the world/base frame.
    - Returns solutions: List of joint configurations, one for each solution. The number of
      solutions is from 0 to 8.
    '''
    
    # Compute the desired transform in the effector's reference frame
    T = dot(T, self.gTe)
    
    # Call the IK solver in universal_robot/ur_kinematics
    result = zeros(48, dtype='float64')
    nSolutions = c_extensions.InverseKinematicsUr5(ascontiguousarray(T), result, 0.0)
    
    # Filter results outside of joint limits.
    solutions = []
    for i in xrange(nSolutions):
      
      solution = result[6 * i: 6 * (i + 1)]      
      
      # rotate by 360 degrees if this can help
      larger2Pi = logical_and(solution > self.jointLimits[1, :],
        solution - 2 * pi > self.jointLimits[0, :])
      solution[larger2Pi] -= 2 * pi
      smaller2Pi = logical_and(solution < self.jointLimits[0, :],
        solution + 2 * pi < self.jointLimits[1, :])
      solution[smaller2Pi] += 2 * pi
      
      # check joint limits
      if self.IsInJointLimits(solution):
        solutions.append(solution)
        
    return solutions
    
  def ClosestIkToGiven(self, configs, target):
    '''Find the joint positions closest to the given joint positions.'''
    
    if isinstance(configs, list): configs = array(configs)
    distance = sum(power(configs - tile(target, (configs.shape[0], 1)), 2), axis = 1)
    return configs[argmin(distance)]
  
  def GetCloud(self, workspace = None):
    '''Renders point cloud from the current sensor position.
    - Input workspace: List of pairs or 2D array of [(xMin, xMax), (yMin, yMax), (zMin, zMax)]
    - Returns cloud: nx3 numpy array.
    '''

    self.StartSensor()
    self.env.StepSimulation(0.001)

    data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
    cloud = data.ranges + data.positions[0]

    self.StopSensor()

    if workspace is not None:
      cloud = point_cloud.FilterWorkspace(workspace, cloud)

    return cloud
    
  def GetCollisionFreeConfigs(self, configs):
    '''Filters given configurations by collisions with the current OpenRAVE environment. Considers
       collisions between arm and gripper with table and all objects currently in the environment.
       Does not consider collisions between the arm and itself. (Gripper and arm?)
    - Input configs: List of arm 6-DoF arm configurations.
    - Returns collisionFreeConfigs: List of the same configurations without those in collision.
    '''
    
    startConfig = self.robot.GetDOFValues()
    
    collisionFreeConfigs = []
    for i, config in enumerate(configs):
      self.MoveRobot(config)
      if not self.env.CheckCollision(self.robot):
        collisionFreeConfigs.append(config)
      
    self.MoveRobot(startConfig)
    return collisionFreeConfigs

  def GetFullCloud(self, viewCenter, viewKeepout, workspace, add45DegViews=False,
    computeNormals=False, voxelSize=None):
    '''Gets a full point cloud of the scene (6 views) and also computes normals.'''

    poses = self.GetFullViewPoses(viewCenter, viewKeepout, add45DegViews)

    cloud = []; viewPoints = []
    for pose in poses:
      self.MoveSensorToPose(pose)
      X = self.GetCloud(workspace)
      cloud.append(X)
      V = tile(pose[0:3, 3], (X.shape[0], 1))
      viewPoints.append(V)

    cloud = vstack(cloud)
    viewPoints = vstack(viewPoints)
    
    if computeNormals:      
      normals = point_cloud.ComputeNormals(cloud, viewPoints, kNeighbors=30, rNeighbors=-1)
      if voxelSize: cloud, normals = point_cloud.Voxelize(voxelSize, cloud, normals)
      return cloud, normals
    
    if voxelSize: cloud = point_cloud.Voxelize(voxelSize, cloud)
    return cloud
    
  def GetFullViewPoses(self, viewCenter, viewKeepout, add45DegViews):
    '''Returns 6 poses, covering the full object. (No face has incidence more than 90 degrees.)'''

    viewPoints = []
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  0,  1]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  0, -1]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  1,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0, -1,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([ 1,  0,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([-1,  0,  0]))

    if add45DegViews:
      viewPoints.append(viewCenter + viewKeepout*array([0, -cos(45*(pi/180)), sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([0,  cos(45*(pi/180)), sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([-cos(45*(pi/180)), 0, sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([cos(45*(pi/180)), 0, sin(45*(pi/180))]))

    upChoice = array([0.9,0.1,0])
    upChoice = upChoice / norm(upChoice)

    viewPoses = []
    for point in viewPoints:
      viewPoses.append(self.GetSensorPoseGivenUp(point, viewCenter, upChoice))

    return viewPoses
    
  def GetHomeConfig(self):
    '''TODO'''
    
    return copy(self.homeConfig)
    
  def GetSensorPoseGivenUp(self, sensorPosition, targetPosition, upAxis):
    '''Generates the sensor pose with the LOS pointing to a target position and the "up" close to a
       specified up.
    - Input sensorPosition: 3-element desired position of sensor placement.
    - Input targetPosition: 3-element position of object required to view.
    - Input upAxis: The direction the sensor up should be close to.
    - Returns T: 4x4 numpy array (transformation matrix) representing desired pose of end effector
      in the base frame.
    '''
  
    v = targetPosition - sensorPosition
    v = v / norm(v)
  
    u = upAxis - dot(upAxis, v) * v
    u = u / norm(u)
  
    t = cross(u, v)
  
    T = eye(4)
    T[0:3,0] = t
    T[0:3,1] = u
    T[0:3,2] = v
    T[0:3,3] = sensorPosition
  
    return T
    
  def GetTableHeight(self):
    '''Accessor method for the table height -- the z position of the top of the table surface.'''
    
    return self.tableHeight
    
  def IsInCollision(self, config):
    '''Check whether the robot is in collision for the given joint positions.'''    
    
    with self.env:
      self.robot.SetDOFValues(config, self.manip.GetArmIndices())
      collision = self.env.CheckCollision(self.robot)
    return collision
    
  def IsInJointLimits(self, config):
    '''Returns true only if the given arm position is within the joint limits.'''
    
    return (config >= self.jointLimits[0, :]).all() and (config <= self.jointLimits[1, :]).all()
    
  def MoveRobot(self, config):
    '''Moves the (simulated) arm to the specified configuration.
    - Input config: Desired 6-DoF configuration of the arm.
    - Returns None.
    '''
    
    with self.env:
      self.robot.SetDOFValues(config, self.manip.GetArmIndices())
    
  def MoveRobotToHome(self):
    '''Moves the arm to the initial configuration specified in the environment XML.'''
    
    self.MoveRobot(self.homeConfig)

  def MoveObjectToHandAtGrasp(self, bTg, objectHandle):
    '''Aligns the grasp on the object to the current hand position and moves the object there.
      - Input: The grasp in the base frame (4x4 homogeneous transform).
      - Input objectHandle: Handle to the object to move.
      - Retruns X: The transform applied to the object.
    '''

    bTo = objectHandle.GetTransform()
    bTr = self.robot.GetLink("gripper_center").GetTransform()

    X = dot(bTr, inv(bTg))
    objectHandle.SetTransform(dot(X, bTo))

    return X
    
  def MoveFloatingHandToPose(self, T):
    '''TODO'''
    
    self.floatingHand.SetTransform(dot(T, self.gcTfh))
    
  def MoveSensorToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    self.sensorRobot.SetTransform(T)

  def PlotCloud(self, cloud, colors = None, pointSize = 0.001):
    '''Plots a cloud in the environment.'''

    if not self.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.UnplotCloud()
      
    if cloud.shape[0] == 0:
      return
      
    if colors is None:
      colors = zeros((cloud.shape[0], 3))
    colors = colors[:, 0:3]
      
    self.plotCloudHandle = self.env.plot3(\
      points = cloud, pointsize = pointSize, colors = colors, drawstyle = 1)

  def PlotDescriptors(self, descriptors, colors = None):
    '''Visualizes grasps in openrave viewer.'''
    
    # input checking    
    if not self.showViewer:
      return

    if self.plotDescriptorsHandle is not None:
      self.UnplotDescriptors()

    if len(descriptors) == 0:
      return
    
    # if 4x4 transform input, convert to HandDescriptor
    descs = []
    for desc in descriptors:
      if not isinstance(desc, HandDescriptor):
        descs.append(HandDescriptor(desc))
      else:
        descs.append(desc)
    descriptors = descs
    
    # if no colors were input, use red
    if colors is None:
      colors = zeros((len(descriptors), 3))
      colors[:, 0] = 1.0
    colors = colors[:, 0:3]

    lineList = []; colorList = []
    for i, desc in enumerate(descriptors):

      c = desc.bottom
      a = c - desc.depth * desc.approach
      l = c - 0.5 * desc.width * desc.binormal
      r = c + 0.5 * desc.width * desc.binormal
      lEnd = l + desc.depth * desc.approach
      rEnd = r + desc.depth * desc.approach

      lineList.append(c); lineList.append(a)
      lineList.append(l); lineList.append(r)
      lineList.append(l); lineList.append(lEnd)
      lineList.append(r); lineList.append(rEnd)
      
      for j in xrange(8):
        colorList.append(colors[i])

    self.plotDescriptorsHandle = self.env.drawlinelist(\
      points = array(lineList), linewidth = 3.0, colors = array(colorList))
      
  def RemoveAttachedObject(self):
    '''TODO'''
    
    if self.attachedObject is not None:
      self.robot.ReleaseAllGrabbed()
      self.env.Remove(self.attachedObject)
      self.attachedObject = None

  def RemoveObjectSet(self, objectHandles):
    '''Removes all of the objects in the list objectHandles.'''

    for objectHandle in objectHandles:
      self.env.Remove(objectHandle)
      
  def RemoveObstacleCloud(self):
    '''TODO'''
    
    if self.obstacleCloud is not None:
      self.env.Remove(self.obstacleCloud)
      self.obstacleCloud = None
      
  def RemoveFloatingHand(self):
    '''TODO'''
    
    if not self.hasFloatingHand: return
    with self.env:
      self.env.Remove(self.floatingHand)
    self.hasFloatingHand = False
      
  def RemoveRobot(self):
    '''Removes the robot arm and gripper from the environment.'''
    
    if not self.hasRobot: return
    with self.env:
      self.env.Remove(self.robot)
    self.hasRobot = False
    
  def RemoveSensor(self):
    '''TODO'''
    
    if not self.hasSensor: return
    with self.env:
      self.env.Remove(self.sensorRobot)
    self.hasSensor = False
      
  def RemoveTable(self):
    '''Removes the table from the environment.'''
    
    if not self.hasTable: return
    with self.env:
      self.env.Remove(self.table)
    self.hasTable = False
    
  def SetHomeConfig(self, config):
    '''TODO'''
    
    self.homeConfig = copy(config)
  
  def SetTableHeight(self, newHeight):
    '''TODO'''

    self.tableHeight = newHeight
    T = self.table.GetTransform()
    T[2, 3] = newHeight
    self.table.SetTransform(T)
        
  def StartSensor(self):
    '''Starts the sensor in openrave, displaying yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOn)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOn)

  def StopSensor(self):
    '''Disables the sensor in openrave, removing the yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)

  def UnplotCloud(self):
    '''Removes a cloud from the environment.'''

    if not self.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.plotCloudHandle.Close()
      self.plotCloudHandle = None

  def UnplotDescriptors(self):
    '''Removes any descriptors drawn in the environment.'''

    if not self.showViewer:
      return

    if self.plotDescriptorsHandle is not None:
      self.plotDescriptorsHandle.Close()
      self.plotDescriptorsHandle = None