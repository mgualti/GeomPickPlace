# scipy
import numpy
from numpy import array, vstack, zeros
# ros
import tf
import rospy
import tf2_ros
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
# self

class CloudProxy:
  '''A class for interfacing with a standard ROS topic point cloud.'''
  
  def __init__(self):
    '''Set variables contained in self.'''
    
    self.hasCloud = True
    
  def InitRos(self):
    '''Call this soon after class initialization but after the ROS node has been initialized.'''
    
    self.activeCloudSub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.Callback,
      queue_size = 1)
    self.cloudPub = rospy.Publisher("/cloud_rviz", PointCloud2, queue_size = 1)
    
    # Wait for nodes to come online.
    print("Waiting for sensor node to connect ...")
    while self.activeCloudSub.get_num_connections() == 0:
      rospy.sleep(1)
      
    # Create TF listener to receive transforms.
    self.tfBuffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tfBuffer)
    self.tfBuffer.lookup_transform("base_link", "ee_link", rospy.Time(0), rospy.Duration(4.0))
  
  def Callback(self, msg):
    '''Called by ROS when a point cloud message has arrived on the subscriber.'''
    
    if not self.hasCloud:
      self.activeCloudMsg = msg
      self.hasCloud = True
  
  def ConvertToPointCloud2(self, cloud, normals = None):
    '''Convert a given numpy array to a point cloud message.'''
    
    header = Header()
    header.frame_id = "base_link"
    header.stamp = rospy.Time.now()
    
    if normals is None:
      return point_cloud2.create_cloud_xyz32(header, cloud)
    
    # concatenate xyz and normals vertically
    pts = zeros((cloud.shape[0],6))
    pts[:,0:3] = cloud[:,0:3]
    pts[:,3:6] = normals[:,0:3]
    
    # create message
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('normal_x', 12, PointField.FLOAT32, 1),
              PointField('normal_y', 16, PointField.FLOAT32, 1),
              PointField('normal_z', 20, PointField.FLOAT32, 1)]
    return point_cloud2.create_cloud(header, fields, pts)  
  
  def GetCloud(self, nFrames = 1):
    '''Wait for a new cloud and convert it into an nx3 array.'''
    
    clouds = []    
    
    for i in xrange(nFrames):
      
      self.hasCloud = False
      while not self.hasCloud:
        rospy.sleep(0.01)
      
      cloudTime = self.activeCloudMsg.header.stamp
      cloudFrame = self.activeCloudMsg.header.frame_id
      cloud = array(list(point_cloud2.read_points(self.activeCloudMsg)))[:,0:3]
      mask = numpy.logical_not(numpy.isnan(cloud).any(axis=1))
      cloud = cloud[mask]
      clouds.append(cloud)
    
    cloud = vstack(clouds)    
    
    print("Received Structure cloud with {} points.".format(cloud.shape[0]))
    return cloud, cloudFrame, cloudTime
  
  def LookupTransform(self, fromFrame, toFrame, lookupTime=rospy.Time(0)):
    '''Lookup a transform in the TF tree.'''
    
    transformMsg = self.tfBuffer.lookup_transform(toFrame, fromFrame, lookupTime, rospy.Duration(1.0))
    
    translation = transformMsg.transform.translation
    pos = [translation.x, translation.y, translation.z]
    rotation = transformMsg.transform.rotation
    quat = [rotation.x, rotation.y, rotation.z, rotation.w]
    T = tf.transformations.quaternion_matrix(quat)
    T[0:3,3] = pos
    return T  
  
  def PlotCloud(self, cloud):
    '''Plots the cloud in rviz.'''
    
    self.cloudPub.publish(self.ConvertToPointCloud2(cloud))
