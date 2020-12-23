#!/usr/bin/env python
'''TODO'''

# python
# scipy
from numpy import array, set_printoptions
# geom_pick_place
from geom_pick_place.environment_blocks import EnvironmentBlocks
# ros
import rospy
# robot
from robot.arm import Arm

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # visualization/saving
  isMoving = True
  showViewer = True
  showWarnings = False
  
  # INITIALIZATION =================================================================================
  
  # initialize environment
  env = EnvironmentBlocks(showViewer, showWarnings)
  env.RemoveFloatingHand()
  env.RemoveSensor()
  
  # initialize ros
  rospy.init_node("GeomPickPlaceRobot")
  
  # initialize arm, gripper, and motion planning
  arm = Arm(env, isMoving)
  
  # wait for ros to catch up
  rospy.sleep(1)
  
  # RUN TEST =======================================================================================
  
  set_printoptions(precision=3)
  while not rospy.is_shutdown():
    print repr(array(arm.jointValues))
    env.MoveRobot(arm.jointValues)
    lessLimits = arm.jointValues < env.jointLimits[0]
    greaterLimits = arm.jointValues < env.jointLimits[0]
    if lessLimits.any(): print("Less limits: {}".format(lessLimits))
    if greaterLimits.any(): print("Greater limits: {}".format(greaterLimits))
    rospy.sleep(2.0)

if __name__ == "__main__":
  main()