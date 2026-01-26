import numpy as np
import torch
import torch.nn.functional as F


def contrastive_centroid(num_classes, class_centroid,weights):
    class_centroid = class_centroid.cuda()
    logis_centroid = torch.einsum('nc,ck->nk',[class_centroid,class_centroid.transpose(0,1)])
    logis_centroid /= 0.5
    centroid_prob = F.normalize(logis_centroid.exp(),dim=1,p=1).log()
    centroid_mask = torch.diag(weights)
    centroid_mask = torch.tensor(centroid_mask).cuda()
    loss_centroid = - torch.sum((centroid_mask * centroid_prob).sum(1)) / centroid_mask.shape[0]
    return loss_centroid

def contrastive_class_to_class_learned_memory( features, class_labels, num_classes, memory, class_centroid):
    """

    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classes in the dataet
        memory: memory bank [List]

    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0

    for c in range(num_classes):
        # get features of an specific class
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        memory_c = memory[c] # N, 256
        class_centroid = class_centroid.cuda()
        if features_c.shape[0] > 1:
            target_mask_c = torch.zeros((num_classes), dtype=float)
            target_mask_c[c] = 1
            target_mask_c = target_mask_c.repeat(features_c.shape[0], 1)
            target_mask_c = target_mask_c.cuda()
            logits = torch.einsum('nc,ck->nk',[F.normalize(features_c,dim=1), class_centroid.transpose(0, 1)])
            logits /= 0.5
            log_prob = F.normalize(logits.exp(),dim=1, p=1).log()
            loss_target = - torch.sum((target_mask_c * log_prob).sum(1)) / target_mask_c.shape[0]
            loss = loss + loss_target

        # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
        # selector = model.__getattr__('contrastive_class_selector_' + str(c))
        # selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()

            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1) # N, 256
            features_c_norm = F.normalize(features_c, dim=1) # M, 256

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            # MxN
            distances = 1 - similarities # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)


            # now weight every sample

            # learned_weights_features = selector(features_c.detach()) # detach for trainability
            # learned_weights_features_memory = selector_memory(memory_c)
            #
            # # self-atention in the memory featuers-axis and on the learning contrsative featuers-axis
            # learned_weights_features = torch.sigmoid(learned_weights_features)
            # # 比平均数大的话就肯定权重大于1，不然就小于1
            # rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)) * learned_weights_features
            # # Mx1=>MxN
            # rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
            # # 逐元素分别相乘
            # distances = distances * rescaled_weights
            #
            #
            # learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            # # Nx1=>1xN
            # learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            # # 比平均数大的话就肯定权重大于1，不然就小于1(这里源代码有错误)
            # rescaled_weights_memory = (learned_weights_features_memory.shape[1] / learned_weights_features_memory.sum(dim=1)) * learned_weights_features_memory
            # # 1xN=>MxN
            # rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            #
            # distances = distances * rescaled_weights_memory


            loss = loss + distances.mean()



    return loss / num_classes



