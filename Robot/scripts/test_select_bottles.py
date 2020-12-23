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
  
  nEpisodes = 30
  nObjects = 2
  
  # bottles = [1,2,5,6,7,8,9,11,12,13,14] # for experiment Type 1 (comparison to previous paper)
  bottles = arange(1, 15) # for experiment Type 2 (comparison to simulation)
  coasters = arange(1, 4)
  
  # RUN TEST =======================================================================================  
  
  print("Bottles: ")
  for i in xrange(nEpisodes):
    objects = sorted(choice(bottles, size = nObjects, replace = False))
    orientation = choice(["u", "s"], size = nObjects, p = [1.0/3.0, 2.0/3.0], replace = True)
    objectString = [str(objects[i]) + orientation[i] for i in xrange(nObjects)]
    objectString = ",".join(objectString)
    print(objectString)
    
  print("Coasters: ")
  for i in xrange(nEpisodes):
    objects = sorted(choice(coasters, size = nObjects, replace = False))
    objectString = [str(objects[i]) for i in xrange(nObjects)]
    objectString = ",".join(objectString)
    print(objectString)
  
if __name__ == "__main__":
  '''Usage: scripts/test_select_objects.py'''
  
  main()