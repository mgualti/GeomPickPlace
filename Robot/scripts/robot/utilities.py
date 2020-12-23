'''A toolbox of routines for moving the arm and performing an experiment.'''

# python
# scipy
from numpy import argsort, linspace, median, ones, sqrt, square
# ros
import rospy
# self
from robot import motion_planner
import point_cloud

class Utilities(object):
  
  def __init__(self, arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner):
    '''TODO'''
    
    # save references to inputs
    
    self.arm = arm
    self.env = arm.env
    self.gripper = gripper
    self.cloudProxy = cloudProxy
    self.taskPlanner = taskPlanner
    self.regraspPlanner = regraspPlanner
    self.motionPlanner = motionPlanner
  
  def GetCloud(self, workspace):
    '''TODO'''
  
    # acquire cloud and filter
    rospy.sleep(1.0)
    cloud, cloudFrame, cloudTime = self.cloudProxy.GetCloud(nFrames = 3)
    cloud = point_cloud.FilterNearAndFarPoints(2, 0.30, 1.00, cloud)
    bTc = self.cloudProxy.LookupTransform(cloudFrame, "base_link", cloudTime)
    cloud = point_cloud.Transform(bTc, cloud)
    cloud = point_cloud.FilterWorkspace(workspace, cloud)
    cloud = point_cloud.Voxelize(0.002, cloud)
  
    # remove table plane
    planeIdxs = point_cloud.SegmentPlane(cloud, 0.017)
    planeIdxs = planeIdxs.flatten()
    mask = ones(cloud.shape[0], dtype = "bool")
    mask[planeIdxs] = False
    plane = cloud[planeIdxs, :]
    cloud = cloud[mask, :]
  
    # estimate table height
    #plane = point_cloud.RemoveStatisticalOutliers(plane, 30, 2)
    tableHeight = median(plane[:, 2]) # median is less sensitive to outliers than mean is
    
    print("Environment table height: {}.".format(self.env.tableHeight))
    print("Median plane height: {}.".format(tableHeight))
    
    cloud[:, 2] += self.env.GetTableHeight() - tableHeight
  
    # return result
    return cloud
    
  def GetConfigs(self, preHand, hand, obstacleCloud):
    '''TODO'''
    
    # compute IK
    preHandConfigs = self.env.CalcIkForT(preHand)
    if len(preHandConfigs) == 0:
      print("No pre-hand configs.")
      return None, None
    
    handConfigs = self.env.CalcIkForT(hand)
    if len(preHandConfigs) == 0:
      print("No hand configs.")
      return None, None
    
    # eliminate solutions in collision
    self.env.AddObstacleCloud(obstacleCloud, 0.01)
    preHandConfigs = self.env.GetCollisionFreeConfigs(preHandConfigs)
    handConfigs = self.env.GetCollisionFreeConfigs(handConfigs)
    self.env.RemoveObstacleCloud()
    
    if len(preHandConfigs) == 0:
      print("All pre-hand configs in collision.")
      return None, None
      
    if len(handConfigs) == 0:
      print("All hand configs in collision.")
      return None, None
      
    # sort solution pairs by distance
    homeConfig = self.env.GetHomeConfig(); distances = []; pairs = []
    for i in xrange(len(preHandConfigs)):
      for j in xrange(len(handConfigs)):
        pairs.append((preHandConfigs[i], handConfigs[j]))
        distances.append(sqrt(sum(square(preHandConfigs[i] - homeConfig))) + 
          sqrt(sum(square(handConfigs[j] - homeConfig))))
    idx = argsort(distances)
    
    # check straight-line motion in order of distance
    for i in xrange(len(pairs)):
      pair = pairs[idx[i]]
      if motion_planner.IsTravelOnLine([pair[0], pair[1]], self.env, 0.015, 0.01):
        return pair[0], pair[1]
    
    print("No pre-hand hand pair is in line.")
    return None, None
    
  def IsObjectInGripper(self):
    '''TODO'''  
    
    # Reports 3 when completely open and 230 when completely closed.  
    
    gripperOpening = self.gripper.GetOpening()
    print("Gripper opening: {}.".format(gripperOpening))
    return gripperOpening < 228
  
  def MoveToConfiguration(self, config):
    '''TODO'''
    
    # generate a motion plan to move the arm
    success, traj = self.motionPlanner.HierarchicalPlan(
      self.arm.GetCurrentConfig(), config, self.env)
    
    if not success:
      print("MoveToConfiguration: Failed to find a motion plan.")
      return False
    
    # Move the arm 
    self.arm.FollowTrajectory(traj, gain = 1.0, gamma = 0.80)
    return True
    
  def VisualizeTrajectory(self, traj, name):
    '''TODO'''
    
    if len(traj) == 0:
      return
    
    self.env.MoveRobot(traj[0])
    raw_input("Starting {}.".format(name))
    
    for i in xrange(1, len(traj)):
      alpha = linspace(0, 1, 60)
      steps = [(1 - alpha[j]) * traj[i - 1] + alpha[j] * traj[i] for j in xrange(len(alpha))]
      for step in steps:
        self.env.MoveRobot(step)
        rospy.sleep(0.05)
    
    raw_input("Finished {}.".format(name))