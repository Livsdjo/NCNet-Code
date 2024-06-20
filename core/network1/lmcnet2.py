import torch.nn as nn
import torch
import numpy as np

from network.knn_search.knn_module import KNN
from network.ops import get_knn_feats, spectral_smooth, trans, compute_smooth_motion_diff
from loss import batch_episym

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None, use_bn=True, use_short_cut=True):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels

        self.use_short_cut=use_short_cut
        if use_short_cut:
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )

    def forward(self, x):
        out = self.conv(x)
        if self.use_short_cut:
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
        return out

class KNNFeats(nn.Module):
    def __init__(self,in_dim, out_dim, downsample=True, downsample_dim=8,use_bn=True):
        super().__init__()
        self.downsample=downsample
        if downsample:
            self.ds_net=nn.Conv2d(in_dim,downsample_dim,1)
            in_dim=downsample_dim

        if use_bn:
            self.mlp=nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
                nn.Conv2d(out_dim, out_dim, 1),
            )
        else:
            self.mlp=nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 1),
            )

    def forward(self, feats, idxs):
        if self.downsample and feats is not None:
            feats=self.ds_net(feats)
        k=idxs.shape[2]
        nn_feats=get_knn_feats(feats,idxs)
        nn_feats_diff=feats.repeat(1,1,1,k)-nn_feats
        nn_feats_out=self.mlp(nn_feats_diff)
        feats_out=torch.max(nn_feats_out, 3, keepdim=True)[0]
        return feats_out

class LCLayer(nn.Module):
    def __init__(self,in_dim,knn_dim):
        super().__init__()
        self.knn_feats=KNNFeats(in_dim,knn_dim,in_dim>knn_dim,knn_dim)
        self.conv_out=nn.Conv2d(knn_dim,in_dim,1)

    def forward(self, feats, idxs):
        feats_knn=self.knn_feats(feats,idxs)
        return self.conv_out(feats_knn)

class CRLayer(nn.Module):
    def __init__(self,eta,feats_dim,eta_learnable=False,use_bn=True):
        super().__init__()
        if use_bn:
            self.filter=nn.Sequential(
                nn.InstanceNorm2d(feats_dim),
                nn.BatchNorm2d(feats_dim),
                nn.ReLU(True),
                nn.Conv2d(feats_dim,feats_dim,1,1),
                nn.InstanceNorm2d(feats_dim),
                nn.BatchNorm2d(feats_dim),
                nn.ReLU(True),
                nn.Conv2d(feats_dim,feats_dim,1,1),
            )
        else:
            self.filter=nn.Sequential(
                nn.InstanceNorm2d(feats_dim),
                nn.ReLU(),
                nn.Conv2d(feats_dim,feats_dim,1,1),
                nn.InstanceNorm2d(feats_dim),
                nn.ReLU(),
                nn.Conv2d(feats_dim,feats_dim,1,1),
            )
        self.eta=eta
        if eta_learnable:
            self.eta=nn.Parameter(torch.from_numpy(np.asarray([self.eta],dtype=np.float32)))

    def forward(self, feats, eig_vec, eig_val,):
        feats_smooth=spectral_smooth(feats,eig_vec,eig_val,self.eta)
        feats_embed=self.filter(feats_smooth-feats)
        return feats_embed

class DiffPool(nn.Module):
    def __init__(self, in_channel, output_points, use_bn=True):
        # in_channel=128, output_points=500
        nn.Module.__init__(self)
        self.output_points = output_points
        if use_bn:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        # x: b,f,n,1
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)  # b,k,n
        # b,f,n @ b,n,k
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out

class DiffUnpool(nn.Module):
    def __init__(self, in_channel, output_points, use_bn=True):
        nn.Module.__init__(self)
        self.output_points = output_points
        if use_bn:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None, use_bn=True):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
                trans(1, 2))
            # Spatial Correlation Layer
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
            )
            self.conv3 = nn.Sequential(
                trans(1, 2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
                trans(1, 2))
            # Spatial Correlation Layer
            self.conv2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
            )
            self.conv3 = nn.Sequential(
                trans(1, 2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class GlobalClusterLayer(nn.Module):
    def __init__(self, in_dim, cluster_num):
        super().__init__()
        self.down = DiffPool(in_dim, cluster_num)
        self.mlp_cluster = OAFilter(in_dim, cluster_num)
        self.up = DiffUnpool(in_dim, cluster_num)

    def forward(self, feats):
        feats_down=self.down(feats)
        feats_down=self.mlp_cluster(feats_down)
        feats_up=self.up(feats,feats_down)
        return feats_up

class LMCBlock(nn.Module):
    def __init__(self, in_dim, knn_dim, eta, eta_learnable, cluster_num):
        super().__init__()
        self.lc0=LCLayer(in_dim, knn_dim)   # 局部特征
        self.cn0=PointCN(in_dim)
        self.cr0=CRLayer(eta, in_dim, eta_learnable)   # 一致性残差层
        self.cluster=GlobalClusterLayer(in_dim, cluster_num)
        self.lc1=LCLayer(in_dim, knn_dim)
        self.cn1=PointCN(in_dim)
        self.cr1=CRLayer(eta, in_dim, eta_learnable)

    def forward(self, feats, idxs, eig_vec, eig_val):
        feats=feats+self.lc0(feats, idxs)
        feats=self.cn0(feats)
        feats=feats+self.cr0(feats, eig_vec, eig_val)
        feats=feats+self.cluster(feats)
        feats=feats+self.lc1(feats, idxs)
        feats=self.cn1(feats)
        feats=feats+self.cr1(feats, eig_vec, eig_val)
        return feats

class LMCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.eta=cfg['eta']
        self.eta=cfg['eta']
        geom_dim=cfg['geom_feats_dim'] # 2 or 0
        self.knn=KNN(cfg['knn_num'])
        self.geom_only=cfg['geom_only']
        if not cfg['geom_only']:
            image_dim=cfg['image_feats_dim'] # 134=128+2+2+2
            self.image_feats_embed=nn.Sequential(
                nn.Conv2d(image_dim,128,1),
                PointCN(128)
            )
        self.geom_feats_embed=nn.Sequential(
            nn.Conv2d(4+2+geom_dim,128,1),
            PointCN(128)
        )
        self.lmcblock_list=nn.ModuleList()
        for k in range(4):
            self.lmcblock_list.append(LMCBlock(128, 8, self.eta, True, 128))

        self.prob_predictor=nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,1,1),
        )

    def forward(self, data):
        res_logits, res_e_hat = [], []
        xs = data['xs'] # b,n,4
        eig_vec = data['eig_vec']
        eig_val = data['eig_val']
        batch_size, num_pts = xs.shape[0], xs.shape[1]
        xs = xs.permute(0,2,1).unsqueeze(3) # b,4,n,1
        _, idxs = self.knn(xs[...,0], xs[...,0])
        idxs = idxs.permute(0, 2, 1)
        motion_diff = compute_smooth_motion_diff(xs,eig_vec,eig_val,self.eta)

        if not self.geom_only:
            image_feats = data['image_feats'] # b,n,f
            image_feats = image_feats.permute(0,2,1).unsqueeze(3) # b,4,n,1
            geom_feats = data['geom_feats'] # b,n,f
            geom_feats = geom_feats.permute(0,2,1).unsqueeze(3) # b,4,n,1
            image_feats = self.image_feats_embed(image_feats)
            geom_feats = self.geom_feats_embed(torch.cat([xs, motion_diff, geom_feats], 1))
            corr_feats = image_feats + geom_feats
        else:
            corr_feats = self.geom_feats_embed(torch.cat([xs, motion_diff], 1))

        for net in self.lmcblock_list:
            corr_feats = net(corr_feats, idxs, eig_vec, eig_val)

        logits=self.prob_predictor(corr_feats) # b,1,n,1

        # logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, logits)
        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        res_logits.append(logits), res_e_hat.append(e_hat)

        return res_logits, res_e_hat
        # return {'logits':logits[:,0,:,0], 'e_hat'}


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    # x = x.permute(2,0,1) # 理解为第零维度用原始第二维度填充，第一维度用原始第零维度填充，第二维度用原始第一维度填充

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat
