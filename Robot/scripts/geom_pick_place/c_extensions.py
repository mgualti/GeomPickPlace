'''Interface to C functions in the extensions folder.'''

import os # python
import ctypes # python
from numpy.ctypeslib import ndpointer # scipy

cextensions = ctypes.cdll.LoadLibrary(os.getcwd() + "/extensions/extensions.so")

IsOccluded = cextensions.IsOccluded
IsOccluded.restype = ctypes.c_int
IsOccluded.argtypes = [\
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_float,
  ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS")]

SearchGraspPlaceGraph = cextensions.SearchGraspPlaceGraph
SearchGraspPlaceGraph.restype = ctypes.c_float
SearchGraspPlaceGraph.argtypes = [\
  ctypes.c_int,
  ctypes.c_int,
  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
  ctypes.c_float,
  ctypes.c_int,
  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
  
InverseKinematicsUr5 = cextensions.InverseKinematicsUr5
InverseKinematicsUr5.restype = ctypes.c_int
InverseKinematicsUr5.argtypes = [\
  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
  ctypes.c_double]