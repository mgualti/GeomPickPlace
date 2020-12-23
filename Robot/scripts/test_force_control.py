#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import array, pi
# geom_pick_place
from geom_pick_place.environment_blocks import EnvironmentBlocks
# ros
import rospy
# robot
from robot.arm import Arm
from robot.utilities import Utilities
from robot.gripper import Gripper
from robot.motion_planner import MotionPlanner

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # motion planning
  homeConfig = array([-0.078, -1.989, 2.243, -1.827, -1.57, -0.023])
  collisionConfig = array([-0.282, -1.136,  2.349, -2.761, -1.569, -0.272])
  maxCSpaceJump = 8 * (pi / 180)
  planningTimeout = 20.0
  
  # visualization/saving
  isMoving = True
  showViewer = True
  showWarnings = False
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveFloatingHand()
  env.RemoveSensor()
  env.SetHomeConfig(homeConfig)
  
  # initialize ros
  rospy.init_node("GeomPickPlaceRobot")
  
  # initialize arm, gripper, and motion planning
  arm = Arm(env, isMoving)
  gripper = Gripper(isMoving)
  motionPlanner = MotionPlanner(maxCSpaceJump, planningTimeout)
  
  # initialize utilities
  utilities = Utilities(arm, gripper, None, None, None, motionPlanner)
  
  # wait for ros to catch up
  rospy.sleep(1)
  
  # RUN TEST =======================================================================================
  
  utilities.MoveToConfiguration(homeConfig)
  gripper.Close()
  
  arm.FollowTrajectory([homeConfig, collisionConfig], gain = 0.4, gamma = 0.70, forceInterrupt = 1000)

if __name__ == "__main__":
  main()