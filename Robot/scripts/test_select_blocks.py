#!/usr/bin/env python
'''For selecting which objects to use during an experiment.'''

# python
import time
# scipy
from numpy.random import choice
from numpy import array, pi
# self


def main():
  '''Entrypoint to test program.'''
  
  # PARAMETERS =====================================================================================
  
  nEpisodes = 10
  nObjects = 5
  
  # Type 1 is 5x5x5, Type 2 is 6x3x3, Type 3 is 6x3x1, Type 4 is 3x3x3
  objectTypes = ['1', '1', '1', '2', '3', '3', '3', '4', '4', '4']
  
  # RUN TEST =======================================================================================  
  
  for i in xrange(nEpisodes):
    objectString = choice(objectTypes, size = nObjects, replace = False)
    objectString = sorted(objectString)
    objectString = ",".join(objectString)
    print(objectString)
  
if __name__ == "__main__":
  '''Usage: scripts/test_select_objects.py'''
  
  main()