import glob
import numpy as np
import random
from random import shuffle
from scipy.io import loadmat

class Data_Configs:
  
  #sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
  #             'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
  #sem_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12]
  #sem_num = len(sem_names)
  
  points_cc = 9
  ins_max_num = 24
  train_pts_num = 4096
  test_pts_num = 4096

class Data:
  
  def __init__(self, configs, dataDirectory, inputName, outputName, train_batch_size):
    
    self.configs = configs
    self.root_folder_4_traintest = dataDirectory
    self.inputName = inputName
    self.outputName = outputName
    self.train_files = self.load_full_file_list()
    #self.test_files = self.load_full_file_list(areas = test_areas)
    print('train files:', len(self.train_files))
    #print('test files:', len(self.test_files))
  
    self.ins_max_num = Data_Configs.ins_max_num
    self.train_batch_size = train_batch_size
    self.total_train_batch_num = len(self.train_files)//self.train_batch_size
  
    self.train_next_bat_index = 0

  def load_full_file_list(self):
    all_files = []
    files = sorted(glob.glob(self.root_folder_4_traintest + '/*.mat'))
    for f in files:
      all_files.append(f)
    return all_files

  def load_raw_data_file_block(self, file_path):
    
    data = loadmat(file_path)
    pc = data[self.inputName]
    ins_labels = data[self.outputName].T
    
    return pc, ins_labels

  def get_bbvert_pmask_labels(self, pc, gt_pmask):
    
    gt_bbvert_padded = np.zeros((self.configs.ins_max_num, 2, 3), dtype = np.float32)

    for i in xrange(self.configs.ins_max_num):
      
      ins_labels_tp_ind = gt_pmask[i, :] > 0.0
      if sum(ins_labels_tp_ind) == 0:
        continue

      ###### bb min_xyz, max_xyz
      pc_xyz_tp = pc[ins_labels_tp_ind]
      gt_bbvert_padded[i, 0, 0] = np.min(pc_xyz_tp[:, 0]) # x_min
      gt_bbvert_padded[i, 0, 1] = np.min(pc_xyz_tp[:, 1]) # y_min
      gt_bbvert_padded[i, 0, 2] = np.min(pc_xyz_tp[:, 2]) # z_min
      gt_bbvert_padded[i, 1, 0] = np.max(pc_xyz_tp[:, 0]) # x_max
      gt_bbvert_padded[i, 1, 1] = np.max(pc_xyz_tp[:, 1]) # y_max
      gt_bbvert_padded[i, 1, 2] = np.max(pc_xyz_tp[:, 2]) # z_max

    return gt_bbvert_padded

  def load_fixed_points(self, file_path):
    
    pc, pmask_padded_labels = self.load_raw_data_file_block(file_path)

    ### center xy within the block
    min_x = np.min(pc[:,0]); max_x = np.max(pc[:,0])
    min_y = np.min(pc[:,1]); max_y = np.max(pc[:,1])
    min_z = np.min(pc[:,2]); max_z = np.max(pc[:,2])

    use_zero_one_center = True
    if use_zero_one_center:
      pc[:, 0:1] = (pc[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
      pc[:, 1:2] = (pc[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
      pc[:, 2:3] = (pc[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)

    bbvert_padded_labels = self.get_bbvert_pmask_labels(pc, pmask_padded_labels)

    return pc, bbvert_padded_labels, pmask_padded_labels

  def load_train_next_batch(self):
    
    bat_files = self.train_files[self.train_next_bat_index * self.train_batch_size : \
      (self.train_next_bat_index + 1) * self.train_batch_size]
    
    bat_pc=[]; bat_bbvert_padded_labels=[]; bat_pmask_padded_labels =[]
    for file in bat_files:
      pc, bbvert_padded_labels, pmask_padded_labels = self.load_fixed_points(file)
      bat_pc.append(pc)
      bat_bbvert_padded_labels.append(bbvert_padded_labels)
      bat_pmask_padded_labels.append(pmask_padded_labels)

    bat_pc = np.asarray(bat_pc, dtype=np.float32)
    bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
    bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

    self.train_next_bat_index += 1
    return bat_pc, bat_bbvert_padded_labels, bat_pmask_padded_labels
  
  def shuffle_train_files(self, ep):
    index = list(range(len(self.train_files)))
    random.seed(ep)
    shuffle(index)
    train_files_new=[]
    for i in index:
        train_files_new.append(self.train_files[i])
    self.train_files = train_files_new
    self.train_next_bat_index=0
    print('train files shuffled!')
