# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import math
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

############################################
#                Path Config               #
############################################
# remoteip = os.popen('pwd').read()
if os.getenv('volna') is not None:
    C.volna = os.environ['volna']
else:
    C.volna = 'D:/code/STPG-main/'
C.repo_name = 'STPG'
C.abs_dir = osp.realpath(".")
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath('log')
C.log_dir_link = osp.join(C.abs_dir, 'log')

# the path for Checkpoints
if os.getenv('snapshot_dir'):
    C.snapshot_dir = osp.join(os.environ['snapshot_dir'], "snapshot")
else:
    C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

# the path for Log
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.train_log_file = C.log_dir + '/train_' + exp_time + '.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

# the path for Data & Weight
C.dataset_path = osp.join(C.volna, 'DATA/pascal_voc')
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.pretrained_model = C.volna + 'DATA/pytorch-weight/resnet50_v1c.pth'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir, 'furnace'))


############################################
#            Experiments Config            #
############################################
C.device = '0'
C.seed = 12345

C.labeled_ratio = 4     # ratio of labeled set
C.nepochs = 800

# Dataset Config
C.train_source = osp.join(C.dataset_path, "subset_train_aug/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(C.dataset_path, "subset_train_aug/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))
C.eval_source = osp.join(C.dataset_path, "val.txt")

# Cutmix Config
C.cutmix_mask_prop_range = (0.25, 0.5)
C.cutmix_boxmask_n_boxes = 3
C.cutmix_boxmask_fixed_aspect_ratio = False
C.cutmix_boxmask_by_size = False
C.cutmix_boxmask_outside_bounds = False
C.cutmix_boxmask_no_invert = False

# Image Config
C.num_classes = 21
C.background = 0
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.image_height = 512
C.image_width = 512
# C.num_train_imgs = 10582 // C.labeled_ratio
C.num_train_imgs = 1464 // C.labeled_ratio
C.num_eval_imgs = 1449
# C.num_unsup_imgs = 0
C.num_unsup_imgs = 10582 - C.num_train_imgs

# Training Config
if os.getenv('learning_rate'):
    C.lr = float(os.environ['learning_rate'])
else:
    C.lr = 0.005

if os.getenv('batch_size'):
    C.batch_size = int(os.environ['batch_size'])
else:
    C.batch_size = 8

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.bn_eps = 1e-5
C.bn_momentum = 0.1
C.unsup_weight = 1

C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)
C.niters_per_epoch = int(math.ceil(C.max_samples * 1.0 // C.batch_size))
C.num_workers = 4
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.warm_up_epoch = 0

# Evaluation Config
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] #[1, 0.75, 1.25]
C.eval_flip = False
C.eval_crop_size = 512

# Display Config
if os.getenv('snapshot_iter'):
    C.snapshot_iter = int(os.environ['snapshot_iter'])
else:
    C.snapshot_iter = 25
C.record_info_iter = 20
C.display_iter = 50