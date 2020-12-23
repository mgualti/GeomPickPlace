import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
from helper_net import Ops
import helper_pointnet2 as pnet2

class BoNet:
  def __init__(self, configs):
    self.points_cc = configs.points_cc
    self.points_num = configs.test_pts_num
    self.bb_num = configs.ins_max_num

  def backbone_pointnet2(self, X_pc, is_train=None):
    
    l0_xyz = X_pc[:,:,0:3]
    l0_points = X_pc[:,:,3:9]
    
    l1_xyz, l1_points, l1_indices = pnet2.pointnet_sa_module(l0_xyz, l0_points, npoint=1024,
      radius=0.1, nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False, is_training=None,
      bn_decay=None, scope='layer1')
    l2_xyz, l2_points, l2_indices = pnet2.pointnet_sa_module(l1_xyz, l1_points, npoint=256,
      radius=0.2, nsample=64, mlp=[64, 64, 128], mlp2=None, group_all=False, is_training=None,
      bn_decay=None, scope='layer2')
    l3_xyz, l3_points, l3_indices = pnet2.pointnet_sa_module(l2_xyz, l2_points, npoint=64,
      radius=0.4, nsample=128, mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=None,
      bn_decay=None, scope='layer3')
    l4_xyz, l4_points, l4_indices = pnet2.pointnet_sa_module(l3_xyz, l3_points, npoint=None,
      radius=None, nsample=None, mlp=[256, 256, 512], mlp2=None, group_all=True, is_training=None,
      bn_decay=None, scope='layer4')

    # Feature Propagation layers
    l3_points = pnet2.pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256],
      is_training=None, bn_decay=None, scope='fa_layer1')
    l2_points = pnet2.pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256],
      is_training=None, bn_decay=None,scope='fa_layer2')
    l1_points = pnet2.pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128],
      is_training=None, bn_decay=None,scope='fa_layer3')
    l0_points = pnet2.pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz, l0_points], axis=-1),
      l1_points, [128, 128, 128, 128], is_training=None, bn_decay=None, scope='fa_layer4')
    global_features = tf.reshape(l4_points, [-1, 512])
    point_features = l0_points

    return point_features, global_features

  ######  2. bbox
  def bbox_net(self, global_features):
    b1 = Ops.xxlu(Ops.fc(global_features, out_d= 512, name='b1'), label='lrelu')
    b2 = Ops.xxlu(Ops.fc(b1, out_d= 256, name='b2'), label='lrelu')

    #### sub branch 1
    b3 = Ops.xxlu(Ops.fc(b2, out_d=256, name='b3'), label='lrelu')
    bbvert = Ops.fc(b3, out_d=self.bb_num * 2 * 3, name='bbvert')
    bbvert = tf.reshape(bbvert, [-1, self.bb_num, 2, 3])
    points_min = tf.reduce_min(bbvert, axis=-2)[:, :, None, :]
    points_max = tf.reduce_max(bbvert, axis=-2)[:, :, None, :]
    y_bbvert_pred = tf.concat([points_min, points_max], axis=-2, name='y_bbvert_pred')

    #### sub branch 2
    b4 = Ops.xxlu(Ops.fc(b2, out_d=256, name='b4'), label='lrelu')
    y_bbscore_pred = tf.sigmoid(Ops.fc(b4, out_d=self.bb_num * 1, name='y_bbscore_pred'))

    return y_bbvert_pred, y_bbscore_pred

  ######  3. pmask
  def pmask_net(self, point_features, global_features, bbox, bboxscore):
    p_f_num = int(point_features.shape[-1])
    p_num = tf.shape(point_features)[1]
    bb_num = int(bbox.shape[1])

    global_features = tf.tile(Ops.xxlu(Ops.fc(global_features, out_d=256, name='down_g1'), 
      label='lrelu')[:,None,None,:], [1, p_num, 1, 1])
    point_features = Ops.xxlu(Ops.conv2d(point_features[:,:,:,None],k=(1, p_f_num), out_c=256,
      str=1, name='down_p1',pad='VALID'), label='lrelu')
    point_features = tf.concat([point_features, global_features], axis=-1)
    point_features = Ops.xxlu(Ops.conv2d(point_features, k=(1,int(point_features.shape[-2])),
      out_c=128, str=1, pad='VALID', name='down_p2'), label='lrelu')
    point_features = Ops.xxlu(Ops.conv2d(point_features, k=(1, int(point_features.shape[-2])),
      out_c=128, str=1, pad='VALID',name='down_p3'), label='lrelu')
    point_features = tf.squeeze(point_features, axis=-2)

    bbox_info = tf.tile(tf.concat([tf.reshape(bbox, [-1, bb_num, 6]), bboxscore[:,:,None]], 
      axis=-1)[:,:,None,:], [1,1,p_num,1])
    pmask0 = tf.tile(point_features[:,None,:,:], [1, bb_num, 1, 1])
    pmask0 = tf.concat([pmask0, bbox_info], axis=-1)
    pmask0 = tf.reshape(pmask0, [-1, p_num, int(pmask0.shape[-1]), 1])

    pmask1 = Ops.xxlu(Ops.conv2d(pmask0, k=(1,int(pmask0.shape[-2])), out_c=64, str=1, pad='VALID',
       name='pmask1'), label='lrelu')
    pmask2 = Ops.xxlu(Ops.conv2d(pmask1, k=(1, 1), out_c=32, str=1, pad='VALID', name='pmask2'), 
      label='lrelu')
    pmask3 = Ops.conv2d(pmask2, k=(1,1), out_c=1, str=1, pad='VALID', name='pmask3')
    pmask3 = tf.reshape(pmask3, [-1, bb_num, p_num])

    y_pmask_logits = pmask3
    y_pmask_pred = tf.nn.sigmoid(y_pmask_logits, name='y_pmask_pred')

    return y_pmask_pred

  ######
  def build_graph(self, GPU='0'):
    #######   1. define inputs
    self.X_pc = tf.placeholder(shape=[None, None, self.points_cc], dtype=tf.float32, name='X_pc')
    self.Y_bbvert = tf.placeholder(shape=[None, self.bb_num, 2, 3], dtype=tf.float32, name='Y_bbvert')
    self.Y_pmask = tf.placeholder(shape=[None, self.bb_num, None], dtype=tf.float32, name='Y_pmask')
    self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
    self.lr = tf.placeholder(dtype=tf.float32, name='lr')

    #######  2. define networks, losses
    with tf.variable_scope('backbone'):
      self.point_features, self.global_features = self.backbone_pointnet2(self.X_pc, self.is_train)

    with tf.variable_scope('bbox'):
      self.y_bbvert_pred_raw, self.y_bbscore_pred_raw = self.bbox_net(self.global_features)
      #### association, only used for training
      bbox_criteria = 'use_all_ce_l2_iou'
      self.y_bbvert_pred, self.pred_bborder = Ops.bbvert_association(
        self.X_pc,  self.y_bbvert_pred_raw, self.Y_bbvert, label=bbox_criteria)
      self.y_bbscore_pred = Ops.bbscore_association(self.y_bbscore_pred_raw, self.pred_bborder)

      ### loss
      self.bbvert_loss, self.bbvert_loss_l2, self.bbvert_loss_ce, self.bbvert_loss_iou = \
        Ops.get_loss_bbvert(self.X_pc, self.y_bbvert_pred, self.Y_bbvert, label=bbox_criteria)
      self.bbscore_loss = Ops.get_loss_bbscore(self.y_bbscore_pred, self.Y_bbvert)
      self.sum_bbox_vert_loss = tf.summary.scalar('bbvert_loss', self.bbvert_loss)
      self.sum_bbox_vert_loss_l2 = tf.summary.scalar('bbvert_loss_l2', self.bbvert_loss_l2)
      self.sum_bbox_vert_loss_ce = tf.summary.scalar('bbvert_loss_ce', self.bbvert_loss_ce)
      self.sum_bbox_vert_loss_iou = tf.summary.scalar('bbvert_loss_iou', self.bbvert_loss_iou)
      self.sum_bbox_score_loss = tf.summary.scalar('bbscore_loss', self.bbscore_loss)

    with tf.variable_scope('pmask'):
      self.y_pmask_pred = self.pmask_net(self.point_features, self.global_features, \
        self.y_bbvert_pred, self.y_bbscore_pred)

      ### loss
      self.pmask_loss = Ops.get_loss_pmask(self.X_pc, self.y_pmask_pred, self.Y_pmask)
      self.sum_pmask_loss = tf.summary.scalar('pmask_loss', self.pmask_loss)

    with tf.variable_scope('pmask', reuse = True):
      #### during testing, no need to associate, use unordered predictions
      self.y_pmask_pred_raw = self.pmask_net(self.point_features, self.global_features, \
        self.y_bbvert_pred_raw, self.y_bbscore_pred_raw)

    ######   3. define optimizers
    var_backbone = [var for var in tf.trainable_variables() if var.name.startswith('backbone') and \
      not var.name.startswith('backbone/sem')]
    var_sem = [var for var in tf.trainable_variables() if var.name.startswith('backbone/sem')]
    var_bbox = [var for var in tf.trainable_variables() if var.name.startswith('bbox')]
    var_pmask = [var for var in tf.trainable_variables() if var.name.startswith('pmask')]

    end_2_end_loss = self.bbvert_loss + self.bbscore_loss  + self.pmask_loss
    self.optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize( \
      end_2_end_loss, var_list = var_bbox + var_pmask + var_backbone + var_sem)

    ######   4. others
    self.saver = tf.train.Saver(max_to_keep = 1)
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.visible_device_list = GPU
    self.sess = tf.Session(config=config)
    self.sess.run(tf.global_variables_initializer())

    return 0
    
  def load_session(self, path, deviceId):
    
    ####### 1. networks
    self.X_pc = tf.placeholder(shape=[None, None, self.points_cc], dtype=tf.float32, name = "X_pc")
    self.is_train = tf.placeholder(dtype=tf.bool, name = "is_train")
    with tf.variable_scope("backbone"):
      self.point_features, self.global_features = self.backbone_pointnet2(self.X_pc, self.is_train)
    with tf.variable_scope("bbox"):
      self.y_bbvert_pred_raw, self.y_bbscore_pred_raw = self.bbox_net(self.global_features)
    with tf.variable_scope("pmask"):
      self.y_pmask_pred_raw = self.pmask_net(
        self.point_features, self.global_features, self.y_bbvert_pred_raw, self.y_bbscore_pred_raw)
    
    ####### 2. restore trained model
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.visible_device_list = str(max(deviceId, 0))
    self.sess = tf.Session(config = config)
    tf.train.Saver().restore(self.sess, path)
