#!/usr/bin/env python
'''TODO'''

# python
import os
import shutil
# WordNet
from nltk.corpus import wordnet

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================  

  downloadFolderName = "/home/mgualti/Downloads/"  
  
  #category = "bottle"
  #offset = "02876657"
  #category = "bowl"
  #offset = "02880940"
  #category = "camera"
  #offset = "02942699"
  #category = "can"
  #offset = "02946921"
  #category = "cap"
  #offset = "02954340"
  #category = "clock"
  #offset = "03046257"
  #category = "mug"
  #offset = "03797390"
  #category = "remote"
  #offset = "04074963"
  category = "telephone"
  offset = "04401088"
  
  # RUN TEST =======================================================================================
  
  # Find the folder name in ShapeNetCore corresponding to the category name.
  
  synset = wordnet.synsets(category)
  
  for s in synset:
    print s.offset()
    
  # Extract .obj files
  
  modelFolderNames = os.listdir(downloadFolderName + offset)
  
  if os.path.exists(downloadFolderName + category):
    shutil.rmtree(downloadFolderName + category)
  os.mkdir(downloadFolderName + category)
  
  for i, folderName in enumerate(modelFolderNames):
    print i, folderName
    shutil.copyfile(
      downloadFolderName + offset + "/" + folderName + "/models/model_normalized.obj",
      downloadFolderName + category + "/" + str(i) + ".obj")
  

if __name__ == "__main__":
  main()