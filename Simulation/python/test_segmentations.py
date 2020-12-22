#!/usr/bin/env python
'''TODO'''

# python
import os
import sys
import fnmatch
from time import time
# scipy
from scipy.io import loadmat, savemat
from numpy import all, repeat, reshape, sum, tile, zeros
from scipy.optimize import linear_sum_assignment
# tensorflow
# self
from bonet.bonet_model import BonetModel

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  scenario = "test1"
  deviceId = cwd = os.getcwd()
  deviceId = 1 if "GeomPickPlace2" in cwd else (0 if "GeomPickPlace1" in cwd else -1)
  saveFileName = "segmentation-results-" + scenario + ".mat"
  
  # segmentations
  nCategories = 6
  nInputPoints = 2048
  scoreThresh = [0.00, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
  
  # objects
  segmentationDirectory = "/home/mgualti/Data/GeomPickPlace/packing_segmentations_" + scenario
  modelFileName = "bonet_model_packing.cptk"

  # INITIALIZATION =================================================================================
  
  # initialize model
  model = BonetModel(nInputPoints, nCategories, deviceId, modelFileName)

  # RUN TEST =======================================================================================
  
  fileNames = os.listdir(segmentationDirectory)
  fileNames = fnmatch.filter(fileNames, "*.mat")
  
  precision = []; recall = []; totalTime = []
  
  for thresh in scoreThresh:
  
    totalPrecision = 0.00; totalRecall = 0.00; startTime = time()
    for i, fileName in enumerate(fileNames):
      
      # load data
      data = loadmat(segmentationDirectory + "/" + fileName)
      I = data["cloud"]
      T = data["segmentation"]
      
      # predict segmentation mask
      P, _, _ = model.PredictRaw(I, thresh)
      
      # evaluate segmentation
      p, r = Evaluate(T, P)
      
      # print progress
      totalPrecision += p; avgPrecision = totalPrecision / (i + 1)
      totalRecall += r; avgRecall = totalRecall / (i + 1)
      avgTime = (time() - startTime) / (i + 1)
      remTime = (len(fileNames) - (i + 1)) * avgTime / 3600
      
      sys.stdout.write("\rfile: {}/{} precision: {}, recall: {}, time: {} hours".format(
        i + 1, len(fileNames), avgPrecision, avgRecall, remTime)); sys.stdout.flush()
    
    sys.stdout.write("\n"); sys.stdout.flush()
    precision.append(avgPrecision); recall.append(avgRecall); totalTime.append(time() - startTime)
    
  data = {"scoreThresh":scoreThresh,"precision":precision, "recall":recall,"totalTime":totalTime}
  savemat(saveFileName, data)

def Evaluate(groundTruth, predicted):
  '''TODO'''
  
  # input checking
  if groundTruth.shape != predicted.shape:
    raise Exception("Ground truth shape {} does not match predicted shape {}.".format( \
      groundTruth.shape, predicted.shape))
  nCategories = groundTruth.shape[1]
  
  # ignore points with no prediction
  idx = predicted.any(axis = 1)
  nPoints = sum(idx)
  falseNegatives = predicted.shape[0] - nPoints
  if nPoints == 0: return 0.0, 0.0
  
  T = groundTruth[idx, :].astype('bool')
  P = predicted[idx, :].astype('bool')
  
  # count mismatches between ground truth and predicted
  a = tile(T, (1, nCategories))
  b = repeat(P, nCategories, axis = 1)
  c = sum((a != b).astype('float32'), axis = 0)
  cost = reshape(c, [nCategories, nCategories])
  
  # find optimal assignment of ground truth to predicted categories
  Tnew = zeros(T.shape, 'bool')
  _, colIdx = linear_sum_assignment(cost)
  for i in xrange(nCategories):
    Tnew[:, i] = T[:, colIdx[i]]
  T = Tnew
    
  # compute precision
  truePositives = sum(all(T == P, axis = 1).astype('float32'), axis = 0)
  precision = truePositives / nPoints
  
  # compute recall
  recall = truePositives / (truePositives + falseNegatives)
  
  # return result
  return precision, recall

if __name__ == "__main__":
  main()