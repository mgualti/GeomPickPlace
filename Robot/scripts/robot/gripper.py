'''This module is for controlling the gripper hardware.'''

import rospy
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input

class Gripper:

  def __init__(self, isMoving):
    '''Constructor.'''

    self.gripperSub = rospy.Subscriber('/Robotiq2FGripperRobotInput', Robotiq2FGripper_robot_input,
      self.UpdateGripperStat)
    self.gripperPub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output,
      queue_size=1)
        
    self.status = None
    self.isMoving = isMoving
    
    print("Waiting for gripper driver to connect ...")
    while self.gripperPub.get_num_connections() == 0 or self.status is None:
      rospy.sleep(0.01)

  def Close(self, speed=255, force=255):
    '''Close the gripper. Default values for optional arguments are set to their max.'''

    if not self.isMoving: return

    print("Closing gripper ...")
    cmd = Robotiq2FGripper_robot_output()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rPR = 255 # position
    cmd.rFR = force
    cmd.rSP = speed
    self.gripperPub.publish(cmd)
    rospy.sleep(0.5)
    
  def GetOpening(self):
    '''TODO'''
    
    return self.status.gPO

  def Open(self, speed=255, force=255):
    '''Open the gripper. Default values for optional arguments are set to their max.'''

    if not self.isMoving: return

    print("Opening gripper ...")
    
    cmd = Robotiq2FGripper_robot_output()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rPR = 0 # position
    cmd.rFR = force
    cmd.rSP = speed
    self.gripperPub.publish(cmd)
    rospy.sleep(0.5)
    
  def UpdateGripperStat(self, msg):
    '''Obtain the status of the gripper.'''

    self.status = msg
