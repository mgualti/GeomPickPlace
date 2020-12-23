#!/usr/bin/env python
'''For selecting which objects to use during an experiment.'''

# python
# scipy
from numpy.random import choice
from numpy import arange
# self


def main():
  '''Entrypoint to test program.'''
  
  # PARAMETERS =====================================================================================
  
  nEpisodes = 10
  nObjects = 6
  
  objectIds = arange(1, 35)
  
  # RUN TEST =======================================================================================  
  
  for i in xrange(nEpisodes):
    objects = sorted(choice(objectIds, size = nObjects, replace = False))
    objects = [str(objects[i]) for i in xrange(nObjects)]
    objectString = ",".join(objects)
    print(objectString)
    
if __name__ == "__main__":
  '''Usage: scripts/test_select_objects.py'''
  
  main()