#!/usr/bin/env python
'''Broadcasts the transform for the sensor.

Credits:
- http://wiki.ros.org/tf/Tutorials/Adding%20a%20frame%20%28Python%29

Assembled: Northeastern University, 2015
'''

import rospy
import tf

if __name__ == '__main__':
  
  rospy.init_node('add_sensor_frame')
  br = tf.TransformBroadcaster()
  rate = rospy.Rate(30)
  
  while not rospy.is_shutdown():
    
    # stand
    br.sendTransform((0.675, 0.062, 0.741), (-0.70710678, 0.00, 0.70710678, 0.00),
      rospy.Time.now(), "camera_link", "base_link")
    
    rate.sleep()
