'''A toolbox of routines for moving the arm and performing an experiment.'''

# python
# scipy
# ros
import rospy
# self
from robot.utilities import Utilities

class UtilitiesBlocks(Utilities):
  
  def __init__(self, arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner):
    
    Utilities.__init__(self, arm, gripper, cloudProxy, taskPlanner, regraspPlanner, motionPlanner)
    
  def ExecutePickPlace(self, homeToPreGraspTraj, graspConfig, homeToPrePlaceTraj, placeConfig):
    '''TODO'''
    
    # home -> preGrasp
    self.arm.FollowTrajectory(homeToPreGraspTraj, gain = 0.60, gamma = 0.90)
    rospy.sleep(1.0)
    
    # preGrasp -> grasp
    self.arm.FollowTrajectory([graspConfig], gain = 0.60, gamma = 0.90, maxDistToTarget = 0.01)
    rospy.sleep(1.0)
    
    # close gripper
    self.gripper.Close(speed = 255, force = 100)
    rospy.sleep(1.0)
    
    # grasp -> preGrasp
    self.arm.FollowTrajectory(homeToPreGraspTraj[-1:], gain = 0.60, gamma = 0.90)
    rospy.sleep(1.0)
    
    # preGrasp -> home
    self.arm.FollowTrajectory(homeToPreGraspTraj[::-1], gain = 0.60, gamma = 0.90)
    rospy.sleep(1.0)
    
    # exit early if the grasp failed
    if not self.IsObjectInGripper():
      return False
    
    # home -> prePlace
    self.arm.FollowTrajectory(homeToPrePlaceTraj, gain = 0.60, gamma = 0.90)
    rospy.sleep(1.0)  
    
    # prePlace -> place
    self.arm.FollowTrajectory([placeConfig], forceInterrupt = 2500, maxDistToTarget = 0.01,
      gain = 0.5, gamma = 0.80)
    rospy.sleep(1.0)
    
    # open gripper
    self.gripper.Open(speed = 255, force = 100)
    rospy.sleep(1.0)
    
    # place -> prePlace
    self.arm.FollowTrajectory(homeToPrePlaceTraj[-1:], gain = 0.60, gamma = 0.90)
    rospy.sleep(1.0)   
    
    # prePlace -> home
    self.arm.FollowTrajectory(homeToPrePlaceTraj[::-1], gain = 0.60, gamma = 0.90)
    rospy.sleep(1.0)   
    
    # cleanup
    return True
    
  def PlanPickPlace(self, grasp, place, obstacleCloudAtFinalPlacement, cloudsAtStart, targObjIdx, showSteps):
    '''TODO'''
    
    # compute grasp IKs
    preGrasp = self.regraspPlanner.GetPreGrasps([grasp.T])[0]
    preGraspConfig, graspConfig = self.GetConfigs(self.env, preGrasp, grasp.T,
      self.regraspPlanner.StackClouds(cloudsAtStart, [targObjIdx]))
    
    if graspConfig is None:
      print("PlanPickPlace: Failed to find (pre-) grasp configuration.")
      return None, None, None, None
      
    # compute place IKs
    prePlace = self.regraspPlanner.GetPreGrasps([place.T])[0]
    prePlaceConfig, placeConfig = self.GetConfigs(
      self.env, prePlace, place.T, obstacleCloudAtFinalPlacement)
    
    if placeConfig is None:
      print("PlanPickPlace: Failed to find (pre-) place configuration.")
      return None, None, None, None
      
    # plan home -> preGrasp
    self.env.AttachObject(cloudsAtStart[targObjIdx], graspConfig, 0.01)
    self.env.AddObstacleCloud(self.regraspPlanner.StackClouds(cloudsAtStart, [targObjIdx]), 0.01)  
    
    success, homeToPreGraspTraj = self.motionPlanner.HierarchicalPlan(
      self.env.GetHomeConfig(), preGraspConfig, self.env)
    
    if not success:
      print("PickAndPlace: Failed to find a motion plan from home to pre-grasp.")
      self.env.RemoveObstacleCloud()
      self.env.RemoveAttachedObject()
      return None, None, None, None
      
    if showSteps:
      self.VisualizeTrajectory(self.env, homeToPreGraspTraj, "home->preGrasp")
      
    # plan home -> prePlace
    self.env.AddObstacleCloud(obstacleCloudAtFinalPlacement, 0.01)
    success, homeToPrePlaceTraj = self.motionPlanner.HierarchicalPlan(
      self.env.GetHomeConfig(), prePlaceConfig, self.env)
    
    if not success:
      print("PickAndPlace: Failed to find a motion plan from home to pre-place.")
      self.env.RemoveObstacleCloud()
      self.env.RemoveAttachedObject()
      return None, None, None, None
    
    if showSteps:
      self.VisualizeTrajectory(self.env, homeToPrePlaceTraj, "home->prePlace")
      
    self.env.RemoveObstacleCloud()
    self.env.RemoveAttachedObject()
    return homeToPreGraspTraj, graspConfig, homeToPrePlaceTraj, placeConfig