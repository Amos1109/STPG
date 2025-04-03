from __future__ import division


import os

import sys
import time
import argparse

from tqdm import tqdm
import numpy as np 
from sklearn.metrics import confusion_matrix
from loss import CrossEntropyLoss2dPixelWiseWeighted
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import CityScape

from furnace.utils.init_func import init_weight, group_weight
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.feature_memory import *
from furnace.utils.contrastive_losses import contrastive_class_to_class_learned_memory
from furnace.engine.engine import Engine

from furnace.seg_opr.loss_opr import ProbOhemCrossEntropy2d
if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False
os.environ["CUDA_VISIBLE_DEVICES"] = config.device




def create_ema_model(model):
    """

    Args:
        model: segmentation model to copy parameters from
        net_class: segmentation model class

    Returns: Segmentation model from [net_class] with same parameters than [model]

    """
    ema_model = Network(config.num_classes, norm_layer=nn.BatchNorm2d)

    for param in ema_model.branch1.parameters():
        param.detach_()
    mp = list(model.branch1.parameters())
    mcp = list(ema_model.branch1.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()


    for param in ema_model.branch2.parameters():
        param.detach_()
    mp = list(model.branch2.parameters())
    mcp = list(ema_model.branch2.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher):
    """

    Args:
        ema_model: model to update
        model: model from which to update parameters
        alpha_teacher: value for weighting the ema_model
        iteration: current iteration

    Returns: ema_model, with parameters updated follwoing the exponential moving average of [model]

    """
    # Use the "true" average until the exponential average is more correct
    # alpha_teacher = min(1 - 1 / (iteration*10 + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model

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
    unsupervised_train_loader_0 = get_train_loader(engine, CityScape, train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1 = get_train_loader(engine, CityScape, train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)

    # Criterion
    pixel_num = 50000 * config.batch_size
    criterion_sup = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=True)
    # criterion_unsup = nn.CrossEntropyLoss(reduction='none', ignore_index=255)
    criterion_unsup = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=255).cuda()

    # Model
    model = Network(config.num_classes, pretrained_model=config.pretrained_model, norm_layer=nn.BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')



    # Optimizer
    params_list_1 = []
    params_list_1 = group_weight(params_list_1, model.branch1.backbone, nn.BatchNorm2d, config.lr)
    for module in model.branch1.business_layer:
        params_list_1 = group_weight(params_list_1, module, nn.BatchNorm2d, config.lr)
    optimizer_1 = torch.optim.SGD(params_list_1, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    params_list_2 = []
    params_list_2 = group_weight(params_list_2, model.branch2.backbone, nn.BatchNorm2d, config.lr)
    for module in model.branch2.business_layer:
        params_list_2 = group_weight(params_list_2, module, nn.BatchNorm2d, config.lr)
    optimizer_2 = torch.optim.SGD(params_list_2, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # LearningRate Policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    ema_model = create_ema_model(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ema_model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer_1=optimizer_1, optimizer_2=optimizer_2)
    if engine.continue_state_object: engine.restore_checkpoint()

    # Init
    # label = [i for i in range(19)]

    ############################################
    #              Begin Training              #
    ############################################
    model.train()
    ema_model.train()
    print ("-----------------------------------------------------------")
    print ('Start Training... ...')
    num_classes = config.num_classes
    label = [i for i in range(19)]

    feature_memory = FeatureMemory(num_samples = config.num_train_imgs, dataset='cityscapes', memory_per_class = 256, feature_size=256, n_classes= num_classes)

    for epoch in range(engine.state.epoch, config.nepochs):

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        # Initialize Dataloader
        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        "Training"
        for idx in pbar:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            # Load the data
            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)

            # Generate the mixed images
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

            "Supervised Part"
            labeled_features, sup_pred_1 = model(imgs, step=1)
            _, sup_pred_2 = model(imgs, step=2)
            loss_sup_1 = criterion_sup(sup_pred_1, gts)
            loss_sup_2 = criterion_sup(sup_pred_2, gts)
            loss_sup = loss_sup_1 + loss_sup_2
            "Unsupervised Part"
            # Estimate the pseudo-label
            with torch.no_grad():
                # teacher#1
                _, logits_u0_tea_1 = ema_model(unsup_imgs_0, step=1)
                _, logits_u1_tea_1 = ema_model(unsup_imgs_1, step=1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # teacher#2
                _, logits_u0_tea_2 = ema_model(unsup_imgs_0, step=2)
                _, logits_u1_tea_2 = ema_model(unsup_imgs_1, step=2)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()
            # Mix teacher predictions using same mask
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            max_prob1, ps_label_1 = torch.max(F.softmax(logits_cons_tea_1, dim=1), dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            max_prob2, ps_label_2 = torch.max(F.softmax(logits_cons_tea_2, dim=1), dim=1)
            ps_label_2 = ps_label_2.long()



            # Get student prediction for mixed image
            # student#1
            unlabeled_features, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
            _, pred_unlabeled_stu1 = torch.max(F.softmax(logits_cons_stu_1, dim=1), dim=1)
            pred_unlabeled_stu1 = pred_unlabeled_stu1.long()
            # loss_w_1 = torch.sum(max_prob2.ge(0.968).long() == 1).item() / np.size(np.array(ps_label_2.cpu()))
            loss_w_1 = max_prob2.detach()
            # student#2
            _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)
            _, pred_unlabeled_stu2= torch.max(F.softmax(logits_cons_stu_2, dim=1), dim=1)
            pred_unlabeled_stu2 = pred_unlabeled_stu2.long()

            confusion = confusion_matrix(np.concatenate([ps_label_1.view(-1).cpu().numpy(),np.array(label)]),np.concatenate([pred_unlabeled_stu2.view(-1).cpu().numpy(),np.array(label)]))
            w = (np.sum(confusion, axis=0) - np.diag(confusion)) / np.sum(confusion, axis=0) + (
                        np.sum(confusion, axis=1) - np.diag(confusion)) / np.sum(confusion, axis=1)
            w = torch.from_numpy(w/np.sum(w)).cuda(non_blocking=True)

            conflict_mask_2 = (pred_unlabeled_stu2 == ps_label_1).float()+(w[ps_label_1] > w[pred_unlabeled_stu2]).float()
            loss_w_2 = max_prob1.detach() * conflict_mask_2

            # Unsupervised Loss
            loss_unsup_1 = criterion_unsup(logits_cons_stu_1, ps_label_2, loss_w_1)
            loss_unsup_2 = criterion_unsup(logits_cons_stu_2, ps_label_1, loss_w_2)
            loss_unsup = (loss_unsup_1 + loss_unsup_2) * config.unsup_weight

            "Total loss"
            loss = loss_sup + loss_unsup

            if epoch > 5:
                with torch.no_grad():
                    labeled_feature_ema, label_pred_ema = ema_model(imgs, step = 2)
                    prob_pred_ema, label_pred_ema = torch.max(torch.softmax(label_pred_ema,dim=1),dim=1)

                labels_down = nn.functional.interpolate(gts.float().unsqueeze(1),size=(labeled_feature_ema.shape[2], labeled_feature_ema.shape[3]), mode='nearest').squeeze(1)
                label_pred_down = nn.functional.interpolate(label_pred_ema.float().unsqueeze(1),size=(labeled_feature_ema.shape[2], labeled_feature_ema.shape[3]), mode='nearest').squeeze(1)
                prob_pred_down = nn.functional.interpolate(prob_pred_ema.float().unsqueeze(1),size=(labeled_feature_ema.shape[2], labeled_feature_ema.shape[3]), mode='nearest').squeeze(1)
                mask_pred_corr = ((label_pred_down == labels_down).float()*(prob_pred_down>0.95).float()).bool()
                labeled_feature_corr = labeled_feature_ema.permute(0, 2, 3, 1)
                labels_down_corr = labels_down[mask_pred_corr]
                labeled_feature_corr = labeled_feature_corr[mask_pred_corr,...]
                feature_memory.add_features_from_sample_learned( labeled_feature_corr, labels_down_corr, config.batch_size, epoch)
            if epoch > 10:
                "labeled data"
                mask_pred_corr = (labels_down != 255)
                labeled_features_all = labeled_features.permute(0, 2, 3, 1)
                labels_down_all = labels_down[mask_pred_corr]
                labeled_features_all = labeled_features_all [mask_pred_corr, ...]
                loss_contr_labeled = contrastive_class_to_class_learned_memory(labeled_features_all, labels_down_all, num_classes, feature_memory.memory, feature_memory.assign)
                loss = loss + loss_contr_labeled * 0.1
                "unlabeled data"
                ps_label_2_down = nn.functional.interpolate(ps_label_2.float().unsqueeze(1), size=(unlabeled_features.shape[2], unlabeled_features.shape[3]),mode='nearest').squeeze(1)
                unlabeled_features = unlabeled_features.permute(0, 2, 3, 1)
                loss_contr_unlabeled = contrastive_class_to_class_learned_memory(unlabeled_features, ps_label_2_down, num_classes, feature_memory.memory, feature_memory.assign)
                loss = loss + loss_contr_unlabeled * 0.1
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            optimizer_1.param_groups[0]['lr'] = lr
            optimizer_1.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_1.param_groups)): optimizer_1.param_groups[i]['lr'] = lr
            optimizer_2.param_groups[0]['lr'] = lr
            optimizer_2.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_2.param_groups)): optimizer_2.param_groups[i]['lr'] = lr

            loss.backward()
            optimizer_1.step()
            optimizer_2.step()
            m = 0.99
            ema_model.branch1 = update_ema_variables(ema_model = ema_model.branch1, model = model.branch1, alpha_teacher = m)
            ema_model.branch2 = update_ema_variables(ema_model = ema_model.branch2, model = model.branch2, alpha_teacher = m)

            if epoch > 10:
                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' loss=%.2f  ' % loss.item() \
                            + ' loss_sup_contra=%.2f' % loss_contr_labeled.item() \
                            + ' loss_unsup_contra=%.2f' % loss_contr_unlabeled.item() \
                            + ' loss_unsup_1=%.2f' % loss_unsup_1.item() \
                            + ' loss_unsup_2=%.2f' % loss_unsup_2.item() \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' (sup_1=%.2f' % loss_sup_1.item() \
                            + ' sup_2=%.2f)  ' % loss_sup_2.item()
            else:
                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' loss=%.2f  ' % loss.item() \
                            + ' loss_unsup_1=%.2f' % loss_unsup_1.item() \
                            + ' loss_unsup_2=%.2f' % loss_unsup_2.item() \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' (sup_1=%.2f' % loss_sup_1.item() \
                            + ' sup_2=%.2f)  ' % loss_sup_2.item()

            pbar.set_description(print_str, refresh=False)
            logfile.write(print_str+'\n')
            logfile.flush()

            end_time = time.time()

        # Save the model
        if (epoch > 110) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            np.save('memory.npy', np.array(feature_memory.memory))
            engine.save_and_link_checkpoint(config.snapshot_dir, config.log_dir, config.log_dir_link)

    logfile.close()