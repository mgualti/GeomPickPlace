'''TODO'''

# python
import os
import fnmatch
# scipy
from scipy.io import loadmat, savemat
# openrave
# self
from environment_packing import EnvironmentPacking

class EnvironmentCanonical(EnvironmentPacking):

  def __init__(self, showViewer, showWarnings):
    '''TODO'''
    
    # initialize a packing environment without the box
    EnvironmentPacking.__init__(self, showViewer, showWarnings)    
    self.RemoveBox()
    
  def GenerateInitialScene(self, nObjects, cloudDirectory, workspace, maxPlaceAttempts = 50):
    '''TODO'''
    
    return super(EnvironmentCanonical, self).LoadInitialScene(
      nObjects, cloudDirectory, workspace, maxPlaceAttempts)
      
  def LoadInitialScene(self, sceneDirectory, sceneNumber):
    '''TODO'''
    
    # input checking
    if not isinstance(sceneDirectory, str):
      raise Exception("Expected type str sceneDirectory; got {}.".format(type(sceneDirectory)))
      
    if not os.path.isdir(sceneDirectory):
      raise Exception("Could not find directory {}.".format(sceneDirectory))
      
    if not isinstance(sceneNumber, (int, long)):
      raise Exception("Expected type int sceneNumber; got {}.".format(type(sceneNumber)))
    
    # reset the initial scene    
    self.ResetScene()
    self.MoveRobotToHome()
    
    # load scene description
    sceneNames = os.listdir(sceneDirectory)
    sceneNames = fnmatch.filter(sceneNames, "*.mat")
    sceneNames = sorted(sceneNames)
    sceneName = sceneNames[sceneNumber % len(sceneNames)]
    sceneData = loadmat(sceneDirectory + "/" + sceneName)
    poses = sceneData["poses"]
    cloudFileNames = sceneData["cloudFileNames"]
    
    # load objects and place them in their saved poses
    for i in xrange(len(poses)):
      fullCloudFileName = cloudFileNames[i].encode("ascii").strip()
      idx = fullCloudFileName.rfind("/")
      cloudDirectory = fullCloudFileName[:idx]
      cloudFileName = fullCloudFileName[idx + 1:]
      body = self.LoadObjectFromFullCloudFile(cloudDirectory, cloudFileName, "object-{}".format(i))
      body.SetTransform(poses[i])
    
  def SaveScene(self, fileName):
    '''TODO'''
    
    poses = []; cloudFileNames = []
    
    for obj in self.objects:
      poses.append(obj.GetTransform())
      cloudFileNames.append(obj.cloudFileName)
    
    data = {"poses":poses, "cloudFileNames":cloudFileNames}
    savemat(fileName, data)