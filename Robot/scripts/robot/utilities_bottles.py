'''A toolbox of routines for moving the arm and performing an experiment.'''

# python
# scipy
from numpy import argsort, concatenate, sqrt, square
# ros
import rospy
# self
import point_cloud
import motion_planner
from robot.utilities import Utilities

class UtilitiesBottles(Utilities):
  
  def __init__(self, arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner):
    
    Utilities.__init__(self, arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner)
    
  def ExecutePickPlace(self, homeToPreGraspTraj, preGraspToHomeTraj, graspConfig, \
    homeToPrePlaceTraj, prePlaceToHomeTraj, placeConfig):
    '''TODO'''
    
    # home -> preGrasp
    self.arm.FollowTrajectory(homeToPreGraspTraj, gain = 0.70, gamma = 0.95)
    rospy.sleep(1.0)
    
    # preGrasp -> grasp
    self.arm.FollowTrajectory([graspConfig], gain = 0.70, gamma = 0.95, maxDistToTarget = 0.01)
    rospy.sleep(1.0)
    
    # close gripper
    self.gripper.Close(speed = 255, force = 125)
    rospy.sleep(1.0)
    
    # grasp -> preGrasp
    self.arm.FollowTrajectory(homeToPreGraspTraj[-1:], gain = 0.70, gamma = 0.95)
    rospy.sleep(1.0)
    
    # preGrasp -> home
    self.arm.FollowTrajectory(preGraspToHomeTraj, gain = 0.70, gamma = 0.95)
    rospy.sleep(1.0)
    
    # exit early if the grasp failed
    if not self.IsObjectInGripper():
      return False
    
    # home -> prePlace
    self.arm.FollowTrajectory(homeToPrePlaceTraj, gain = 0.70, gamma = 0.95)
    rospy.sleep(1.0)  
    
    # prePlace -> place
    self.arm.FollowTrajectory([placeConfig], forceInterrupt = 3000, maxDistToTarget = 0.01,
      gain = 0.60, gamma = 0.80)
    rospy.sleep(1.0)
    
    # open gripper
    self.gripper.Open(speed = 255, force = 100)
    rospy.sleep(1.0)
    
    # place -> prePlace
    self.arm.FollowTrajectory(homeToPrePlaceTraj[-1:], gain = 0.70, gamma = 0.95)
    rospy.sleep(1.0)   
    
    # prePlace -> home
    self.arm.FollowTrajectory(prePlaceToHomeTraj, gain = 0.70, gamma = 0.95)
    rospy.sleep(1.0)   
    
    # cleanup
    return True
    
  def GetConfigs(self, preHand, hand, objectCloud, obstacleCloud, attachConfig):
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
    handConfigs = self.env.GetCollisionFreeConfigs(handConfigs)
    
    if len(handConfigs) == 0:
      print("All hand configs in collision.")
      self.env.RemoveObstacleCloud()
      return None, None    
    
    if attachConfig is None: attachConfig = handConfigs[0]
    self.env.AttachObject(objectCloud, attachConfig, 0.01)    
    preHandConfigs = self.env.GetCollisionFreeConfigs(preHandConfigs)
    
    self.env.RemoveAttachedObject()
    self.env.RemoveObstacleCloud()
    
    if len(preHandConfigs) == 0:
      print("All pre-hand configs in collision.")
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
    
  def PlanPickPlace(self, grasp, place, objectTransform, objectCloud, obstacleCloud, showSteps):
    '''TODO'''
    
    # compute grasp IKs
    preGrasp = self.regraspPlanner.GetPreGrasps([grasp.T])[0]
    preGraspConfig, graspConfig = self.GetConfigs(
      preGrasp, grasp.T, objectCloud, obstacleCloud, None)
    
    if graspConfig is None:
      print("PlanPickPlace: Failed to find (pre-) grasp configuration.")
      return None, None, None, None, None, None
      
    # compute place IKs
    prePlace = self.regraspPlanner.GetPreGrasps([place.T])[0]
    prePlaceConfig, placeConfig = self.GetConfigs(
      prePlace, place.T, objectCloud, obstacleCloud, graspConfig)
    
    if placeConfig is None:
      print("PlanPickPlace: Failed to find (pre-) place configuration.")
      return None, None, None, None, None, None
      
    # plan home -> preGrasp
    self.env.AddObstacleCloud(concatenate([objectCloud, obstacleCloud]), 0.01)
    success, homeToPreGraspTraj = self.motionPlanner.HierarchicalPlan(
      self.env.GetHomeConfig(), preGraspConfig, self.env)
    
    if not success:
      print("PickAndPlace: Failed to find a motion plan from home to pre-grasp.")
      self.env.RemoveObstacleCloud()
      return None, None, None, None, None, None
      
    '''if showSteps:
      self.VisualizeTrajectory(homeToPreGraspTraj, "home->preGrasp")'''
      
    # plan preGrasp -> home
    self.env.AddObstacleCloud(obstacleCloud, 0.01)
    self.env.AttachObject(objectCloud, graspConfig, 0.01)
    success, preGraspToHomeTraj = self.motionPlanner.HierarchicalPlan(
      preGraspConfig, self.env.GetHomeConfig(), self.env)
    
    if not success:
      print("PickAndPlace: Failed to find a motion plan from pre-grasp to home.")
      self.env.RemoveObstacleCloud()
      self.env.RemoveAttachedObject()
      return None, None, None, None, None, None
      
    '''if showSteps:
      self.VisualizeTrajectory(homeToPreGraspTraj, "preGrasp->home")'''
      
    # plan home -> prePlace
    success, homeToPrePlaceTraj = self.motionPlanner.HierarchicalPlan(
      self.env.GetHomeConfig(), prePlaceConfig, self.env)
    
    if not success:
      print("PickAndPlace: Failed to find a motion plan from home to pre-place.")
      self.env.RemoveObstacleCloud()
      self.env.RemoveAttachedObject()
      return None, None, None, None, None, None
    
    '''if showSteps:
      self.VisualizeTrajectory(homeToPrePlaceTraj, "home->prePlace")'''
      
    self.env.RemoveObstacleCloud()
    self.env.RemoveAttachedObject()
    
    # plan prePlace -> home
    obstacleCloudAfterPlacement = concatenate( \
      [point_cloud.Transform(objectTransform, objectCloud), obstacleCloud])
    self.env.AddObstacleCloud(obstacleCloudAfterPlacement, 0.01)
    success, prePlaceToHomeTraj = self.motionPlanner.HierarchicalPlan(
      prePlaceConfig, self.env.GetHomeConfig(), self.env)
      
    if not success:
      print("PickAndPlace: Failed to find a motion plan from pre-place to home.")
      self.env.RemoveObstacleCloud()
      return None, None, None, None, None, None
    
    '''if showSteps:
      self.VisualizeTrajectory(prePlaceToHomeTraj, "prePlace->home")'''
    
    self.env.RemoveObstacleCloud()
    return homeToPreGraspTraj, preGraspToHomeTraj, graspConfig, homeToPrePlaceTraj, \
      prePlaceToHomeTraj, placeConfig