# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_att, target_cam, p):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:5][:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:5][:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:5][:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:5][:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    elif sampler == 'bce_triplet':
        def loss_func(score, feat, target, target_att, target_cam, p):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if isinstance(score, list):
                    bce_wl_loss = torch.nn.BCEWithLogitsLoss()
                    if cfg.MODEL.JPM and cfg.MODEL.ATTR:
                        JPM_LOSS = [F.cross_entropy(scor, target) for scor in score[1:5][:]]
                        JPM_LOSS = sum(JPM_LOSS) / len(JPM_LOSS)
                        ATTR_LOSS = bce_wl_loss(score[5], target_att)
                    elif cfg.MODEL.ATTR:
                        adaptation_factor = 2 / (1+math.exp(-10.0*p)) - 1
                        ATTR_LOSS = [bce_wl_loss(scor, target_att) for scor in score[1:]]
                        ATTR_LOSS = sum(ATTR_LOSS) / len(ATTR_LOSS)
                        ID_LOSS = 1.0 * ATTR_LOSS + 1.0 * F.cross_entropy(score[0], target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                if isinstance(feat, list):
                    if cfg.MODEL.JPM and cfg.MODEL.ATTR:
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    elif cfg.MODEL.ATTR:
                        ATTR_TRI_LOSS = triplet(feat[1], target)[0]
                        adaptation_factor = 2 / (1+math.exp(-10.0*p)) - 1
                        TRI_LOSS = triplet(feat[0], target)[0]
                else:
                    TRI_LOSS = triplet(feat, target)[0]

                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


