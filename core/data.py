from __future__ import print_function
import sys
import h5py
import numpy as np
import cv2
import torch
import torch.utils.data as data
from utils import np_skew_symmetric


def collate_fn(batch):
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())

    data = {}
    data['K1s'], data['K2s'], data['Rs'], \
        data['ts'], data['xs'], data['ys'], data['T1s'], data['T2s'], data['virtPts'], data['sides'], data['images'], data['eig_val'], data['eig_vec'], data['kp_i'], data['kp_j'] = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for sample in batch:
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['T1s'].append(sample['T1'])
        data['T2s'].append(sample['T2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        data['virtPts'].append(sample['virtPt'])
        data['eig_val'].append(sample['eig_val'])
        # data['eig_vec'].append(sample['eig_vec'])
        data['images'].append(sample['images'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp)
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
            data['eig_vec'].append(sample['eig_vec'][sub_idx,:])
            data['kp_i'].append(sample['kp_i'][sub_idx, :])
            data['kp_j'].append(sample['kp_j'][sub_idx, :])
            if sample['side'] != []:
                data['sides'].append(sample['side'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])
            data['kp_i'].append(sample['kp_i'])
            data['kp_j'].append(sample['kp_j'])
            data['eig_vec'].append(sample['eig_vec'])
            if sample['side'] != []:
                data['sides'].append(sample['side'])

    for key in ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s','virtPts', 'eig_val', 'eig_vec']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    if data['sides'] != []:
        data['sides'] = torch.from_numpy(np.stack(data['sides'])).float()
    return data



class CorrespondencesDataset(data.Dataset):
    def __init__(self, filename, config):
        self.config = config
        self.filename = filename
        self.data = None
        self.eig_known = None

    def correctMatches(self, e_gt):
        step = 0.1
        xx,yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
        # Points in first image before projection
        pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
        # Points in second image before projection
        pts2_virt_b = np.float32(pts1_virt_b)
        pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1,-1,2), pts2_virt_b.reshape(1,-1,2)

        pts1_virt_b, pts2_virt_b = cv2.correctMatches(e_gt.reshape(3,3), pts1_virt_b, pts2_virt_b)

        return pts1_virt_b.squeeze(), pts2_virt_b.squeeze()

    def norm_input(self, x):
        x_mean = np.mean(x, axis=0)
        dist = x - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
        return x, T
    
    def __getitem__(self, index):
        # print("index", index)
        # print("111", self.data)
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
            # print(self.filename)
            # print(self.data.keys())
            # ['Rs', 'cx1s', 'cx2s', 'cy1s', 'cy2s', 'eig_data', 'f1s', 'f2s', 'mutuals', 'ratios', 'ts', 'xs', 'ys']

        xs = np.asarray(self.data['xs'][str(index)])
        ys = np.asarray(self.data['ys'][str(index)])
        R = np.asarray(self.data['Rs'][str(index)])
        t = np.asarray(self.data['ts'][str(index)])
        images = np.asarray(self.data['images'][str(index)])
        eig_data = np.asarray(self.data['eig_data'][str(index)])
        kp_i = np.asarray(self.data['kp_i'][str(index)])
        kp_j = np.asarray(self.data['kp_j'][str(index)])
        # print("eig_data", eig_data.shape, self.data['eig_data'][str(index)].shape, type(self.data['eig_data'][str(index)].shape))
        # images = np.asarray(self.data['images'][str(index)])
        
        eig_val, eig_vec = eig_data[0], eig_data[1:]
        # print("101010", xs.shape, eig_val.shape, eig_vec.shape, eig_val)
        
        # print(5555, xs)
        #是否使用后面的傅里叶基以及特征值
        """
        if self.eig_known is None:
            self.eig_known = np.load("/home/featurize/work/69-82.npy")
            eig_val_known, eig_vec_knwon = self.eig_known[0], self.eig_known[1:]
            # print("knwon", eig_val_known.shape, eig_vec_knwon.shape)
        """
        
        
        side = []
        if self.config.use_ratio == 0 and self.config.use_mutual == 0:
            pass
        elif self.config.use_ratio == 1 and self.config.use_mutual == 0:
            mask = np.asarray(self.data['ratios'][str(index)]).reshape(-1) < config.ratio_test_th
            xs = xs[:,mask,:]
            ys = ys[:,mask]
        elif self.config.use_ratio == 0 and self.config.use_mutual == 1:
            mask = np.asarray(self.data['mutuals'][str(index)]).reshape(-1).astype(bool)
            xs = xs[:,mask,:]
            ys = ys[:,mask]
        elif self.config.use_ratio == 2 and self.config.use_mutual == 2:
            side.append(np.asarray(self.data['ratios'][str(index)]).reshape(-1,1)) 
            side.append(np.asarray(self.data['mutuals'][str(index)]).reshape(-1,1))
            side = np.concatenate(side,axis=-1)
        else:
            raise NotImplementedError

        e_gt_unnorm = np.reshape(np.matmul(
            np.reshape(np_skew_symmetric(t.astype('float64').reshape(1,3)), (3, 3)), np.reshape(R.astype('float64'), (3, 3))), (3, 3))
        e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
        
        if self.config.use_fundamental:
            cx1 = np.asarray(self.data['cx1s'][str(index)])
            cy1 = np.asarray(self.data['cy1s'][str(index)])
            cx2 = np.asarray(self.data['cx2s'][str(index)])
            cy2 = np.asarray(self.data['cy2s'][str(index)])
            f1 = np.asarray(self.data['f1s'][str(index)])
            f2 = np.asarray(self.data['f2s'][str(index)])
            K1 = np.asarray([
                [f1[0], 0, cx1[0]],
                [0, f1[1], cy1[0]],
                [0, 0, 1]
                ])
            K2 = np.asarray([
                [f2[0], 0, cx2[0]],
                [0, f2[1], cy2[0]],
                [0, 0, 1]
                ])
            x1, x2 = xs[0,:,:2], xs[0,:,2:4]
            x1 = x1 * np.asarray([K1[0,0], K1[1,1]]) + np.array([K1[0,2], K1[1,2]])
            x2 = x2 * np.asarray([K2[0,0], K2[1,1]]) + np.array([K2[0,2], K2[1,2]])
            # norm input
            x1, T1 = self.norm_input(x1)
            x2, T2 = self.norm_input(x2)
            
            xs = np.concatenate([x1,x2],axis=-1).reshape(1,-1,4)
            # get F
            e_gt = np.matmul(np.matmul(np.linalg.inv(K2).T, e_gt), np.linalg.inv(K1))
            # get F after norm
            e_gt_unnorm = np.matmul(np.matmul(np.linalg.inv(T2).T, e_gt), np.linalg.inv(T1))
            e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
        else:
            K1, K2 = np.zeros(1), np.zeros(1)
            T1, T2 = np.zeros(1), np.zeros(1)

        pts1_virt, pts2_virt = self.correctMatches(e_gt)

        pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')
        
        return {'K1':K1, 'K2':K2, 'R':R, 't':t, 'xs':xs, 'ys':ys, 'T1':T1, 'T2':T2, 'virtPt':pts_virt, 'side':side, 'eig_val':eig_val, 'eig_vec': eig_vec, 'images':images, 'kp_i': kp_i, 'kp_j': kp_j}
        
    def reset(self):
        if self.data is not None:
            self.data.close()
        self.data = None

    def __len__(self):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
            _len = len(self.data['xs'])
            self.data.close()
            self.data = None
        else:
            _len = len(self.data['xs'])
        return _len

    def __del__(self):
        if self.data is not None:
            self.data.close()

