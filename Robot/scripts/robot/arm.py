'''This module is for controlling the arm hardware.'''

# python
# scipy
from numpy.linalg import norm
from numpy import abs, array, max, zeros
# ros
import rospy
from std_msgs.msg import String
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs

class Arm:

  def __init__(self, env, isMoving):
    '''Initializes arm parameters, joint limits, and controller and robot state topics.'''

    # inputs
    self.env = env
    self.isMoving = isMoving

    # parameters
    self.controllerFrameRate = 125.0
    self.acceleration = "1.0" # rad/sec^2

    # initialize joint values
    self.jointValues = [0] * 6
    self.jointVelocities = [0] * 6

    # initialize publishers and subscribers
    self.jointSub = rospy.Subscriber("/joint_states", sensor_msgs.JointState, self.JointsCallback)
    self.forceSub = rospy.Subscriber("/wrench", geometry_msgs.WrenchStamped, self.ForceCallback)
    self.publisher = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size = 10)

    # register callback for CTRL+C
    rospy.on_shutdown(self.ShutdownCallback)

  def FollowTrajectory(self, traj, gain, gamma, forceInterrupt = float('inf'), maxErrorMag = 1.5,
    maxDistToTarget = 0.02):
    '''Simple leaky integrator with error scaling and breaking.'''

    if not self.isMoving:
      print("Skipping followTrajectory since isMoving=False.")
      return

    leakySum = 0.0
    forceInterruptCount = 0
    command = zeros(6)

    print("Moving arm...")
    for i, target in enumerate(traj):
      error = target - self.jointValues

      while max(abs(error)) > maxDistToTarget:

        # check force
        if self.forceMag >= forceInterrupt:
          print("Force magnitude of {} exceeded threshold of {}.".format(
            self.forceMag, forceInterrupt))
          forceInterruptCount += 1
          if forceInterruptCount > 3 * self.controllerFrameRate: break
          error = zeros(6)

        # scale to maximum error
        errorMag = norm(error)
        if errorMag > maxErrorMag:
          scale = maxErrorMag / errorMag
          error = scale * error

        # integrate error
        leakySum = gamma * leakySum + (1.0 - gamma) * error
        command = gain * leakySum

        self.PublishVelocities(command)
        rospy.sleep(1.0 / self.controllerFrameRate)

        # compute error
        error = target - self.jointValues

    print("Breaking...")
    self.PublishStop()

    # set speed to 0.0
    self.PublishVelocities(zeros(6))

  def ForceCallback(self, msg):
    '''Callback function for the joint_states ROS topic.'''

    self.forceMag = msg.wrench.force.x ** 2 + msg.wrench.force.y ** 2 + msg.wrench.force.z ** 2

  def GetCurrentConfig(self):
    '''TODO'''

    return array(self.jointValues)

  def JointsCallback(self, msg):
    '''Callback function for the joint_states ROS topic.'''

    self.jointValues = (msg.position[2], msg.position[1], msg.position[0], msg.position[3], \
      msg.position[4], msg.position[5])
    self.jointVelocities = (msg.velocity[2], msg.velocity[1], msg.velocity[0], msg.velocity[4], \
      msg.velocity[5])
    self.jointNames = msg.name

  def PublishStop(self):
    '''TODO'''

    self.publisher.publish("stop(" + self.acceleration + ")")

  def PublishVelocities(self, velocities):
    '''TODO'''

    self.publisher.publish( \
      "speedj(" + str(velocities.tolist()) + ", " + self.acceleration + ", 0.025)")

  def ShutdownCallback(self):
    '''Gradually reduces the joint velocities to zero when the program is trying to shut down.'''

    print("Received shutdown signal ...")
    self.PublishStop()
