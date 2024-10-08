import torch
import numpy as np

def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii).cuda(), torch.from_numpy(desc_jj).cuda()
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0]).cuda()).cpu().numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).cpu().numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.cpu().numpy()]
    return idx_sort, ratio_test, mutual_nearest

"""
def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = np.array(desc_ii), np.array(desc_jj)
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = np.sqrt(np.expand_dims(d1, 1) + np.expand_dims(d2, 0) - 2*np.matmul(desc_ii, desc_jj.T))
    nnIdx1 = np.argpartition(distmat, kth=1, axis=1)[:,0]
    nnIdx2 = np.argpartition(distmat, kth=1, axis=0)[0]
    mutual_nearest = (nnIdx2[nnIdx1] == np.arange(nnIdx1.shape[0]))
    ratio_test = distmat[np.arange(distmat.shape[0]), nnIdx1] / np.partition(distmat, kth=1, axis=1)[:,1]
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1]
    return idx_sort, ratio_test, mutual_nearest
"""
