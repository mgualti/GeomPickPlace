# GeomPickPlaceRobot

* **Authors:** Marcus Gualteri
* **Version:** 1.0.0


## 1) Overview

This package demonstrates regrasping on the UR5 robot.

## 2) Setup

1. Start Universal_Robots_ROS_Driver driver.

roslaunch ur_robot_driver ur5_bringup.launch kinematics_config:=/home/mgualti/rosws/src/GeomPickPlaceRobot/calibration/mercury.yaml robot_ip:=192.168.0.106

2. Start gripper driver.

rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
  
3. Start depth sensor.

roslaunch openni2_launch openni2.launch

4. Start rviz.

roslaunch GeomPickPlaceRobot GeomPickPlaceRobot.launch

5. Enter correct directory.

cd /home/mgualti/rosws/src/GeomPickPlaceRobot

## 3) Bottles on Coasters

scripts/test_bottles_on_coasters.py

## 4) Commands for testing/configuration

sudo chmod 777 /dev/ttyUSB0
sudo adduser mgualti tty
rosrun robotiq_2f_gripper_control Robotiq2FGripperSimpleController.py
roslaunch ur_calibration calibration_correction.launch robot_ip:=172.22.22.1 target_filename:=mercury_calibration.yaml
