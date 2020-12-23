# scipy
import numpy
from numpy import array, vstack, zeros
# ros
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
# self

class PlotRviz:
  '''A class for interfacing with a standard ROS topic point cloud.'''
  
  def __init__(self):
    '''Set variables contained in self.'''
    
    self.descriptorMarkers = []
    
  def initRos(self):
    '''Call this soon after class initialization but after the ROS node has been initialized.'''
    
    self.descsPub = rospy.Publisher("/descriptors", Marker, queue_size=1)
    
    rospy.on_shutdown(self.shutdownCallback)
  
  def plotDescriptors(self, descriptors, color=[1,1,1]):
    '''Plots each HandDescriptor in the descriptors list. Uses the given rgb color in [0,1].'''
    
    if len(descriptors) == 0:
      return
    
    marker = CreateMarker("descriptors", len(self.descriptorMarkers), rgb=color, alpha=0.5)    
    
    for desc in descriptors:

      c = desc.bottom
      a = c - desc.depth*desc.approach
      l = c - 0.5*desc.width*desc.binormal
      r = c + 0.5*desc.width*desc.binormal
      lEnd = l + desc.depth*desc.approach
      rEnd = r + desc.depth*desc.approach

      marker.type = Marker.LINE_LIST
  
      c = desc.bottom
      a = c - desc.depth*desc.approach
      l = c - 0.5*desc.width*desc.binormal
      r = c + 0.5*desc.width*desc.binormal
      lEnd = l + desc.depth*desc.approach
      rEnd = r + desc.depth*desc.approach
  
      # approach line  
      marker.points.append(Point(c[0], c[1], c[2]))
      marker.points.append(Point(a[0], a[1], a[2]))
      
      # base line
      marker.points.append(Point(l[0], l[1], l[2]))
      marker.points.append(Point(r[0], r[1], r[2]))
  
      # left finger
      marker.points.append(Point(l[0], l[1], l[2]))
      marker.points.append(Point(lEnd[0], lEnd[1], lEnd[2]))
      
      # right finger
      marker.points.append(Point(r[0], r[1], r[2]))
      marker.points.append(Point(rEnd[0], rEnd[1], rEnd[2]))  
  
      marker.scale.x = marker.scale.y = marker.scale.z = desc.height

    self.descriptorMarkers.append(marker)
    self.descsPub.publish(marker)
  
  def shutdownCallback(self):
    '''Called when ROS is shutting down.'''
    
    self.unplotDescriptors()
  
  def unplotDescriptors(self):
    '''Deletes all of the currently plotted descriptors.'''
    
    for marker in self.descriptorMarkers:
      marker.action = Marker.DELETE
      self.descsPub.publish(marker)
    
    self.descriptorMarkers = []
    
    
def CreateMarker(topicName, id, tfFrame="base_link", lifetime=rospy.Duration(0), rgb=[1,0,0], alpha=1):
  '''Create a visual marker.'''
  
  marker = Marker()
  marker.ns = topicName
  marker.id = id
  marker.header.frame_id = tfFrame  
  marker.lifetime = lifetime
  marker.color.r = rgb[0]
  marker.color.g = rgb[1]
  marker.color.b = rgb[2]
  marker.color.a = alpha
  marker.action = Marker.ADD
  marker.header.stamp = rospy.get_rostime()
  
  return marker