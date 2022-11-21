import torch
import numpy as np
import os
from config import cfg
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded. (sc)
        Key: for each query identity, its gallery images from the same camera view and clothes are discarded. (cc)
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    
    for q_idx in range(num_q):
        # get query pid, camid, and clothid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_clothid = q_clothids[q_idx]

        order = indices[q_idx]  # select one row
        if cfg.DATASETS.NAMES == "nkup":
            # remove gallery samples that have the same pid and same camid with query
            # remove gallery samples that have the same pid and same clothid with query
            if cfg.MODEL.Evaluate == "ClothChangingSetting":
                remove1 = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                remove2 = (g_pids[order] == q_pid) & (g_clothids[order] == q_clothid)
                remove = remove1 + remove2

            # remove gallery samples that have the same pid and same camid with query
            # remove gallery samples that have the same pid and different clothid with query
            elif cfg.MODEL.Evaluate == "StandardSetting":
                remove1 = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                remove2 = (g_pids[order] == q_pid) & (g_clothids[order] != q_clothid)
                remove = remove1 + remove2
        elif cfg.DATASETS.NAMES == "ilrw":
            # remove gallery samples that have the same pid and same clothid with query
            if cfg.MODEL.Evaluate == "ClothChangingSetting":
                remove = (g_pids[order] == q_pid) & (g_clothids[order] == q_clothid)

            # remove gallery samples that have the same pid and different clothid with query
            elif cfg.MODEL.Evaluate == "StandardSetting":
                remove = (g_pids[order] == q_pid) & (g_clothids[order] != q_clothid)
        elif cfg.DATASETS.NAMES == "prcc":
            if cfg.MODEL.Evaluate == "ClothChangingSetting":
                remove1 = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                remove2 = (g_pids[order] == q_pid) & (g_clothids[order] == q_clothid)
                remove = remove1 + remove2
            elif cfg.MODEL.Evaluate == "StandardSetting":
                remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]     ### updated
        # y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0              ### original
        # tmp_cmc = tmp_cmc / y                                     ### original
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query      # 493
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.clothids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, clothid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.clothids.extend(np.asarray(clothid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)    # feats.shape: [6512, 1536] with attributes or [6512, 768] without attributes
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query] # qf.shape: [493, 768]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_clothids= np.asarray(self.clothids[:self.num_query])
        # gallery
        gf = feats[self.num_query:] # gf.shape: [6019, 768]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_clothids = np.asarray(self.clothids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)   # original paper's parameter values
            # distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)    # distmat.shape: (493, 6019)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids)

        return cmc, mAP