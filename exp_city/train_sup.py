from __future__ import division


import os

import sys
import time
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import CityScape

from furnace.utils.init_func import init_weight, group_weight
from furnace.engine.lr_policy import WarmUpPolyLR

from furnace.engine.engine import Engine

from furnace.seg_opr.loss_opr import ProbOhemCrossEntropy2d
if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False
os.environ["CUDA_VISIBLE_DEVICES"] = config.device




############################################
#                  CutMix                  #
############################################
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                           n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size,
                                           within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)
add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(mask_generator)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)


############################################
#                   Main                   #
############################################
def main():
    logfile = open(config.train_log_file, 'a')
    parser = argparse.ArgumentParser()
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True

        seed = config.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Dataloader (Sup & Unsup)
        train_loader = get_train_loader(engine, CityScape, train_source=config.train_source, unsupervised=False, collate_fn=collate_fn)

        # Criterion
        pixel_num = 50000 * config.batch_size
        criterion_sup = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=True)

        # Model
        model = Network(config.num_classes, pretrained_model=config.pretrained_model, norm_layer=nn.BatchNorm2d)
        init_weight(model.branch1.business_layer, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')



        # Optimizer
        params_list_1 = []
        params_list_1 = group_weight(params_list_1, model.branch1.backbone, nn.BatchNorm2d, config.lr)
        for module in model.branch1.business_layer:
            params_list_1 = group_weight(params_list_1, module, nn.BatchNorm2d, config.lr)
        optimizer_1 = torch.optim.SGD(params_list_1, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

        # Optimizer
        params_list_2 = []
        params_list_2 = group_weight(params_list_2, model.branch2.backbone, nn.BatchNorm2d, config.lr)
        for module in model.branch2.business_layer:
            params_list_2 = group_weight(params_list_2, module, nn.BatchNorm2d, config.lr)
        optimizer_2 = torch.optim.SGD(params_list_2, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

        # LearningRate Policy
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        engine.register_state(dataloader=train_loader, model=model, optimizer_1=optimizer_1,optimizer_2=optimizer_2)
        if engine.continue_state_object: engine.restore_checkpoint()

        # Init
        # label = [i for i in range(19)]

        ############################################
        #              Begin Training              #
        ############################################
        model.train()
        print ("-----------------------------------------------------------")
        print ('Start Training... ...')
        for epoch in range(engine.state.epoch, config.nepochs):

            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            if is_debug:
                pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
            else:
                pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

            # Initialize Dataloader
            dataloader = iter(train_loader)

            "Training"
            for idx in pbar:
                optimizer_1.zero_grad()
                engine.update_iteration(epoch, idx)
                start_time = time.time()

                # Load the data
                minibatch = dataloader.next()

                imgs = minibatch['data']
                gts = minibatch['label']

                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)


                "Supervised Part"
                _, sup_pred_1 = model(imgs,step=1)
                loss_sup = criterion_sup(sup_pred_1, gts) * config.unsup_weight

                current_idx = epoch * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)
                optimizer_1.param_groups[0]['lr'] = lr
                optimizer_1.param_groups[1]['lr'] = lr
                for i in range(2, len(optimizer_1.param_groups)): optimizer_1.param_groups[i]['lr'] = lr

                loss_sup.backward()
                optimizer_1.step()


                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' loss=%.2f  ' % loss_sup.item() \

                pbar.set_description(print_str, refresh=False)
                logfile.write(print_str+'\n')
                logfile.flush()

                end_time = time.time()

            # Save the model
            if (epoch > config.nepochs // 6) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
                engine.save_and_link_checkpoint(config.snapshot_dir, config.log_dir, config.log_dir_link)

        logfile.close()

if __name__== "__main__" :
    main()