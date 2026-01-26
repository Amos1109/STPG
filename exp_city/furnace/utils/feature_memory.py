"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import random

class FeatureMemory:
    def __init__(self, num_samples, dataset,  memory_per_class=2048, feature_size=256, n_classes=19):
        self.num_samples = num_samples
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        self.class_centroid = F.normalize(torch.randn(n_classes,feature_size))
        self.assign = F.normalize(torch.randn(n_classes,feature_size))
        if dataset == 'cityscapes': # usually all classes in one image
            self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
            self.target = np.load('./optimal_city_19_256.npy')
            self.target = torch.tensor(self.target)
        elif dataset == 'pascal_voc': # usually only around 3 classes on each image, except background class
            self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))
            self.target = np.load('./optimal_voc_21_256.npy')
            self.target = torch.tensor(self.target)



    def add_features_from_sample_learned(self,  features, class_labels, batch_size, epoch):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        elements_per_class = batch_size * self.per_class_samples_per_image

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            # selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
            # 这就是一个二维矩阵
            features_c = features[mask_c, :] # get features from class c
            if features_c.shape[0] > 0:
                features_centriod = F.normalize(torch.mean(F.normalize(features_c, dim=1),dim=0),dim=0).cpu()
                self.class_centroid[c] = 0.9 * self.class_centroid[c] + 0.1 * features_centriod
                self.class_centroid[c] = F.normalize(self.class_centroid[c], dim=0)
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        # rank = selector(features_c)
                        # rank = torch.sigmoid(rank)
                        if epoch > 10:
                            feature_target = self.assign[c].cuda()
                            rank = torch.mm(F.normalize(features_c, dim=1), torch.unsqueeze(feature_target, 1))  # MxN
                            # sort them
                            _, indices = torch.sort(rank[:, 0], dim=0, descending=True)
                            indices = indices.cpu().numpy()
                            features_c = features_c.cpu().numpy()
                            # get features with highest rankings
                            features_c = features_c[indices, :]
                            new_features = features_c[:elements_per_class, :]
                        else:
                            new_features = features_c.cpu().numpy()[:elements_per_class, :]
                else:
                    new_features = features_c.cpu().numpy()
                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]

        centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.target.transpose(0,1)])
        centroid_target_dist = centroid_target_dist.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(- centroid_target_dist)

        for one_label, one_idx in zip(row_ind, col_ind):
            # if iter % 3000 == 0 and iter != 0:
            #     print((one_label,one_idx))
            self.assign[one_label] = self.target[one_idx]