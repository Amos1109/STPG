#!/usr/bin/env python3
# encoding: utf-8
import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import dataloader

from config import config
from furnace.utils.pyt_utils import ensure_dir, parse_devices
from furnace.utils.visualize import print_iou
from furnace.engine.evaluator import Evaluator
from furnace.engine.logger import get_logger
from furnace.seg_opr.metric import hist_info, compute_score

from dataloader import VOC
from dataloader import ValPre
from network import Network
from multiprocessing.reduction import ForkingPickler

logger = get_logger()
default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

def get_class_colors(*args):
    def uint82bin(n, count=8):
        """
        returns the binary of integer n, count refers to amount of bits
        """
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    N = 21
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors[0:]


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            'save colored result'
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            'save raw result'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        print ("----------------------------")
        print("class_nums: ", len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc, dataset.get_class_names(), True)

        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = VOC(data_setting, 'val', val_pre, training=False)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset,
                                 config.num_classes,
                                 config.image_mean,
                                 config.image_std,
                                 network,
                                 config.eval_scale_array,
                                 config.eval_flip,
                                 all_dev,
                                 args.verbose,
                                 args.save_path,
                                 args.show_image)

        segmentor.run(config.snapshot_dir,
                      args.epochs,
                      config.val_log_file,
                      config.link_val_log_file)
