'''Provides a class for representing a hand pose and a hand volume.'''

# python
# openrave
# scipy
from numpy import copy, cross, eye
# self
import point_cloud

class HandDescriptor():

  def __init__(self, T, height = 0.01, width = 0.085, depth = 0.065):
    '''Creates a HandDescriptor object with everything needed.
    - Input depth: For the Robotiq 85 gripper, we measured 0.070 when open and 0.075 when closed.
    '''

    self.T = T

    # hand size
    self.height = height
    self.width = width
    self.depth = depth

    # hand axes
    self.axis =      T[0:3, 0]
    self.binormal =  T[0:3, 1]
    self.approach = -T[0:3, 2]
    self.center =    T[0:3, 3]
    self.bottom = self.center - 0.5 * self.depth * self.approach
    self.top = self.center + 0.5 * self.depth * self.approach
    
    
  def GetPointsInHandFrame(self, cloud):
    '''TODO'''
    
    # === a bit of a hack for the real robot, where the descriptor size used in training does
    # not match the modeled descriptor size
    handClosingRegion = [(-0.010 / 2, 0.010 / 2), (-0.085 / 2, 0.085  / 2), (-0.075  / 2, 0.075  / 2)]
    T = copy(self.T)
    T[0:3, 3] = T[0:3, 3] + (self.depth / 2 - handClosingRegion[2][1]) * T[0:3, 2]
    # ===    
    
    X = point_cloud.Transform(point_cloud.InverseTransform(T), cloud)
    return point_cloud.FilterWorkspace(handClosingRegion, X)

# UTILITIES ========================================================================================

def DescriptorsFromPoses(poses):
    '''TODO'''
    
    descriptors = []
    for T in poses:
      descriptors.append(HandDescriptor(T))
    return descriptors

def FlipDescriptor(descriptor):
  '''TODO'''
  
  T = eye(4)
  T[0:3, 0] = -descriptor.T[0:3, 0]
  T[0:3, 1] = -descriptor.T[0:3, 1]
  T[0:3, 2] =  descriptor.T[0:3, 2]
  T[0:3, 3] =  descriptor.T[0:3, 3]
  
  return HandDescriptor(T)

def PoseFromApproachAxisCenter(approach, axis, center):
  '''Given grasp approach and axis unit vectors, and center, get homogeneous transform for grasp.'''

  T = eye(4)
  T[0:3, 0] = axis
  T[0:3, 1] = cross(-approach, axis)
  T[0:3, 2] = -approach
  T[0:3, 3] = center

  return T
  
def PoseFromApproachBinormalCenter(approach, binormal, center):
  '''Given grasp approach and closing unit vectors, and center, get homogeneous transform for grasp.'''

  T = eye(4)
  T[0:3, 0] = cross(binormal, -approach)
  T[0:3, 1] = binormal
  T[0:3, 2] = -approach
  T[0:3, 3] = center

  return T