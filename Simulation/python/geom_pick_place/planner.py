'''A set of tools common to the regrasp planner and the task planners.'''

# python
# scipy
from numpy.linalg import norm
from scipy.special import erf
from scipy.spatial import cKDTree, ConvexHull
from numpy import abs, argmax, argmin, copy, cross, empty, eye, logical_and, mean, ones, sqrt, \
  reshape, sum, tile, vstack
# self
import point_cloud
from hand_descriptor import HandDescriptor
from environment_pick_place import EnvironmentPickPlace

class Planner(object):

  def __init__(self, env, minPointsPerSegment):
    '''Instantiate Planner base class.
    
    - Input env: An EnvironmentPickPlace instance.
    - Input minPointsPerSegment: Used by SegmentObjects: a segment must have this many points to be
      considered an object.
    '''
    
    # input checking
    if not isinstance(env, EnvironmentPickPlace):
      raise Exception("Expected env to derive from (or be an instance of) EnvironmentPickPlace.")
      
    if not isinstance(minPointsPerSegment, (int, long)):
      raise Exception("Expected minPointsPerSegment to be an integer.")
      
    if minPointsPerSegment <= 0:
      raise Exception("minPointsPerSegment (={}) must be positive.".format(minPointsPerSegment))
    
    # assignments
    self.env = env
    self.minPointsPerSegment = minPointsPerSegment
    
    # initialization
    self.InitializeHandRegions()
    
  def CompleteObjects(self, clouds, model, useGroundTruth):
    '''Predict the complete shape of objects, given partial views.
    
    - Input clouds: List of n_i x 3 point clouds.
    - Input model: Instance of PcnModel.
    - Input useGroundTruth: If True, ignores model and samples points from the object's mesh.
      (Only valid if the point cloud was simulated using an instance of Environment.) If the given
      segmentation is erroneous, the association of segments to ground truth objects will be
      ambiguous. Thus, this should usually be used with ground truth segmentation.
    - Returns completions: A completed cloud (n_j x 3 numpy array) for each input cloud.
    - Returns compCorrectProbs: A list of correctness probabilities, for each input cloud, a list of
      n_j-element vectors indicating the probability each point is completed "correctly".
    '''
    
    completions = []; compCorrectProbs = []
    
    if useGroundTruth:
      
      # preprocessing
      modelTrees = []
      for obj in self.env.unplacedObjects:
        modelCloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
        modelTrees.append(cKDTree(modelCloud))
      
      for cloud in clouds:
        
        # figure out which model corresponds to this segmentation
        distances = []
        for tree in modelTrees:
          d, _ = tree.query(cloud)
          distances.append(mean(d))
          
        # sample points on the full cloud for this model
        modelCloud = modelTrees[argmin(distances)].data
        completions.append(copy(modelCloud))
        compCorrectProbs.append(ones(modelCloud.shape[0]))
    
    else:
      
      # use PCN to predict each completed shape
      for cloud in clouds:
        completion, correctProbs = model.Predict(cloud, self.env.GetTableHeight())
        completions.append(completion)
        compCorrectProbs.append(correctProbs)
      
    return completions, compCorrectProbs
  
  def GetNearestOrthogonalUnitVector(self, vector, target):
    '''Returns the unit vector orthogonal to vector and nearest target.
    
    - Input vector: 3-element numpy array. Assumes this is a (not necessarily unit) nonzero vector.
    - Input target: 3-element numpy array. Assumes this is a (not necessarily unit) nonzero vector.
    - Returns result: normalized 3-element numpy array.
    '''
    
    # input checking
    if vector.size != 3:
      raise Exception("vector must be a 3-element numpy array.")
    
    if target.size != 3:
      raise Exception("target must be a 3-element numpy array.")
    
    # nearest orthogonal unit vector
    result = cross(vector, cross(target, vector))
    return result / norm(result)
  
  def GetOrthogonalUnitVector(self, vector):
    '''Returns a unit vector orthogonal to the provided vector.
    
    - Input vector: 3-element numpy array. Assumes this is a (not necessarily unit) nonzero vector.
    - Returns result: normalized 3-element numpy array.
    '''
    
    # input checking
    
    if vector.size != 3:
      raise Exception("vector must be a 3-element numpy array.")

    # compute an orthogonal unit vector    

    idx = argmax(abs(vector))
    result = ones(3, dtype = vector.dtype)
    
    if idx == 0:
      result[0] = - (vector[1] + vector[2]) / vector[0]
    elif idx == 1:
      result[1] = - (vector[0] + vector[2]) / vector[1]
    else:
      result[2] = - (vector[0] + vector[1]) / vector[2]
      
    return result / norm(result)
    
  def GetSupportSurfaces(self, cloud):
    '''Finds facets on the object which, when aligned on a horizontal support surface, will result
       in a stable placement of the object. For a precise definition of stable placements, see
       "Regrasping" by Tournassound, Lozano-Perez, and Mazer. For the ray-triangle intersection
       method used, see http://geomalgorithms.com/a06-_intersect-2.html. This is a preprocessing
       step for sampling stable placements.
       
      - Input cloud: Point cloud (nx3 numpy array) of the object to find support surfaces for.
      - Returns supportingTriangles: mx3 integer array, indexing cloud, indicating which points
        belong to a supporting facet (a triangle).
      - Returns triangleNormals: Outward-facing unit normal vector (mx3 numpy array) for each
        supporting triangle.
      - Returns center: The object's center of mass -- the average postion of the input cloud.
      '''
    
    # Input checking
    
    if cloud.shape[1] != 3:
      raise Exception("cloud must be 3D.")
    
    # Compute convex hull of the point cloud.
    
    hull = ConvexHull(cloud)
    triangles = hull.simplices
    #print("Found {} triangles.".format(triangles.shape[0]))
      
    # Determine if the line from CoM along the facet normal direction interesects the facet.
    
    # compute normals for each traingle
    triVect1 = cloud[triangles[:, 1], :] - cloud[triangles[:, 0], :]
    triVect2 = cloud[triangles[:, 2], :] - cloud[triangles[:, 0], :]
    triangleNormals = cross(triVect1, triVect2)
      
    # compute points of intersection from CoM to planes formed by triangles
    center = mean(cloud, axis = 0) # the cloud CoM
    triPoint0 = cloud[triangles[:, 0], :] # a point on the plane
    distToPlane = sum(triangleNormals * (triPoint0 - center), axis = 1)
    pI = tile(center, (triangleNormals.shape[0], 1)) + tile(reshape(distToPlane, 
      (triangleNormals.shape[0], 1)), (1, 3)) * triangleNormals
      
    # determine if these points lie in the triangles
    w = pI - triPoint0
    uu = sum(triVect1 * triVect1, axis = 1)
    uv = sum(triVect1 * triVect2, axis = 1)
    vv = sum(triVect2 * triVect2, axis = 1)
    wu = sum(w * triVect1, axis = 1)
    wv = sum(w * triVect2, axis = 1)
    denom = (uv * uv - uu * vv)
    # remove faces which consist of colinear points ===
    if (denom == 0).any():
      idx = denom != 0
      triangles = triangles[idx]; triangleNormals = triangleNormals[idx]
      distToPlane = distToPlane[idx]; w = w[idx]; uu = uu[idx]; uv = uv[idx]; vv = vv[idx]
      wu = wu[idx]; wv = wv[idx]; denom = denom[idx]
    # ===
    sI = (uv * wv - vv * wu) / denom
    tI = (uv * wu - uu * wv) / denom
    
    inTriangle = logical_and(logical_and(sI >= 0, tI >= 0), sI + tI <= 1)
    supportingTriangles = triangles[inTriangle]
    triangleNormals = triangleNormals[inTriangle]
    distToPlane = distToPlane[inTriangle]
    #print("Found {} supporting triangles.".format(supportingTriangles.shape[0]))
    
    # Normalize normals.
    lenTriangleNormals = norm(triangleNormals, axis = 1)
    # remove faces which consist of colinear points ===
    if (lenTriangleNormals == 0).any():
      idx = lenTriangleNormals != 0
      supportingTriangles = supportingTriangles[idx]
      triangleNormals = triangleNormals[idx]
      lenTriangleNormals = lenTriangleNormals[idx]
      distToPlane = distToPlane[idx]
    # ===
    triangleNormals /= tile(reshape(lenTriangleNormals, (triangleNormals.shape[0], 1)), (1, 3))
    
    # Face normals outward.
    inwardNormals = distToPlane < 0
    triangleNormals[inwardNormals] = -triangleNormals[inwardNormals]
    
    return supportingTriangles, triangleNormals, center
    
  def InCollisionWithHandElements(self, cloudInHandFrame):
    '''Checks intersection of point cloud with simplified, rectangular hand model. The hand model is
       determined by the dimensions in HandDescriptor.
    
    - Input cloudInHandFrame: nx3 numpy array of points with respect to the HandDescriptor frame.
    - Returns True if at least one point intersects the hand model and False otherwise. 
    '''
    
    X = point_cloud.FilterWorkspace(self.handFingerRegionL, cloudInHandFrame)
    if X.size > 0: return True
    X = point_cloud.FilterWorkspace(self.handFingerRegionR, cloudInHandFrame)
    if X.size > 0: return True
    X = point_cloud.FilterWorkspace(self.handTopRegion, cloudInHandFrame)
    if X.size > 0: return True
    return False
      
  def InitializeHandRegions(self):
    '''Determines geometry of a simplified, rectangular hand model, in the hand's reference frame.
    Used by InCollisionWithHandElements for fast collision with point cloud estimation. Only needs
    to be called once during initialization.
    '''
    
    # find default descriptor geometry
    desc = HandDescriptor(eye(4))
    
    # cuboids representing hand regions, [(minX, maxX), (minY, maxY), (minZ, maxZ)]
    self.handClosingRegion = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width  / 2, desc.width  / 2),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handFingerRegionL = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width / 2 - 0.01, -desc.width / 2),
      (-desc.depth / 2, desc.depth / 2)]
      
    self.handFingerRegionR = [
      (-desc.height / 2, desc.height / 2),
      (desc.width / 2, desc.width / 2 + 0.01),
      (-desc.depth / 2, desc.depth / 2)]
      
    self.handTopRegion = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width / 2 - 0.01, desc.width / 2 + 0.01),
      (desc.depth / 2, desc.depth / 2 + 0.01)]
      
  def NormalCdf(self, x, mu = 0.0, sigma = 1.0):
    '''Cumulative distribution function for a normal distribution.
    
    - Input x: Input to the CDF: Pr(X <= x).
    - Input mu: Mean of the distribution.
    - Input sigma: Standard deviation of the distribution.
    - Returns: Probability a sample is less than or equal to x.
    '''
    
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))
      
  def SegmentObjects(self, cloud, model, scoreThresh, useGroundTruth):
    '''Given a point cloud of a scene, segments each object instance.
    
    - Input cloud: Points acquired from the depth sensor: an nx3 numpy array.
    - Input model: BoNet model trained for object instance segmentation.
    - Input scoreThresh: No prediction is made for points having segmentation correctness
      probability below this threshold. See the precision recall curve for the trained model for
      deciding this threshold, or use <= 0 to require a prediction for all points (i.e., all points
      in the downsampled, fixed-size cloud).
    - Input useGroundTruth: If True, uses the models in the Environment to get a perfect
      segmentation. In this case, BoNet is only used to determine the number of object categories.
      (This, of course, only produces a valid result for clouds simulated from Environment.)
    - Returns objects: An m-element list of point clouds corresponding to predicted object instances
      (each an n_i x 3 numpy array). Each point is a member of the original cloud, but all points in
      the original cloud are not necessarily represented unless scoreThresh <= 0 and
      minPointsPerSegment = 1.
    - Returns probs: An m-element list of n_i-element vectors, predicting the probability each point
      in object i is correctly segmented.
    - Returns distribution: An n x nCategories matrix, where each row gives the probability the
      corresponding point belongs to each category. (Note the rows are not necessarily normalized
      due to a loose interpretation of BoNet's output mask as probabilities.)
    '''
    
    # input checking
    if cloud.shape[1] != 3:
      raise Exception("Expected cloud to be 3D (is {}D).".format(cloud.shape[1]))
      
    if not isinstance(useGroundTruth, bool):
      raise Exception("Expected bool for useGroundTruth (is {})".format(type(useGroundTruth)))
    
    # segment objects
    if useGroundTruth:
      segmentation = self.env.GetSegmentation(cloud, model.nCategories)
      segCorrectProbs = ones(segmentation.shape[0])
      distribution = segmentation.astype("float")
    else:
      segmentation, segCorrectProbs, distribution = model.Predict(cloud, scoreThresh)
    
    # convert segmentation mask into a list of objects: filter objects by minimum number of points
    objects = []; probs = []
    for i in xrange(model.nCategories):
      obj = cloud[segmentation[:, i], :]
      if obj.shape[0] >= self.minPointsPerSegment:
        objects.append(obj)
        probs.append(segCorrectProbs[segmentation[:, i]])
    
    return objects, probs, distribution
      
  def StackClouds(self, clouds, skipIndices = []):
    '''Concatenates a list of point clouds, excluding indicated clouds.
    
    - Input clouds: m-element list of n_i x 3 numpy arrays.
    - Input skipIndices: List of indices in [0, ..., m-1] indicating which clouds to not include
      in the concatenation.
    - Returns concatenated clouds.
    '''
    
    stack = []
    for i, cloud in enumerate(clouds):
      if i not in skipIndices:
        stack.append(cloud)
    
    return empty((0, 3)) if len(stack) == 0 else vstack(stack)
    
  def TruncatedNormalCdf(self, x, mu, sigma, a, b):
    '''Cumulative distribution function for truncated normal distribution. See
       https://en.wikipedia.org/wiki/Truncated_normal_distribution.
      
    - Input x: The input to the CDF: Pr(X <= x).
    - Input mu: Offset parameter (the mode if a <= mu <= b).
    - Input sigma: Spread parameter (like standard deviation).
    - Input a: Minimum value a random variable can take.
    - Input b: Maximum value a random variable can take.
    - Returns: Probability a sample is less than x.
    '''
    
    if b <= a: raise Exception("b must be greater than a.")
    if sigma <= 0: raise Exception("sigma must be positive.")
    
    epsilon = (x - mu) / sigma; alpha = (a - mu) / sigma; beta = (b - mu) / sigma;
    return (self.NormalCdf(epsilon) - self.NormalCdf(alpha)) / \
      (self.NormalCdf(beta) - self.NormalCdf(alpha))