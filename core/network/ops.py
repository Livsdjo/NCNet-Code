import torch
import torch.nn as nn
from network.pointnet2_ext.pointnet2_module import grouping_operation

class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

    

# 计算 lambda^k 的和
def sum_of_powers(lambdas, k, xishu):
    result = 0
    for i in range(k):
        result += torch.pow(lambdas, k) * xishu[i]
    # return torch.sum(torch.pow(lambdas, k))
    return result

# 构造对角矩阵
def diag_matrix(lambdas, k):
    diag_elements = [sum_of_powers(lambdas[i], k, [1, 2, 3]) if i == j else 0 for i in range(len(lambdas)) for j in range(len(lambdas))]
    # diag_matrix = torch.tensor(diag_elements).reshape(len(lambdas), len(lambdas))
    return diag_matrix   


def get_knn_feats(feats,idxs):
    """
    :param feats:  b,f,n,1  float32
    :param idxs:   b,n,k    int32
    :return: b,f,n,k
    """
    return grouping_operation(feats[...,0].contiguous(),idxs.int().contiguous()) # b,f,n,k

def spectral_smooth(feats, eig_vec, eig_val, eta):
    """
    :param feats:    b,f,n,1
    :param eig_vec:  b,n,eig_dim
    :param eig_val:  b,eig_dim
    :param eta:      float
    :return:         b,f,n,1
    """
    # print("spectral", feats.shape, feats[...,0].shape)
    feats_proj = torch.bmm(feats[...,0],eig_vec) # b,f,eig_dim
    # 傅里叶系数 torch.Size([16, 128, 32]) torch.Size([16, 32])
    A = torch.pow(feats_proj, 2)
    B = eig_val.unsqueeze(2)
    erci_xing = torch.bmm(A, B)
    erci_xing_loss = torch.norm(erci_xing, p=2, dim=(1, 2))  # p='fro'
    # print("A__erci", feats_proj.shape, A.shape, erci_xing.shape, erci_xing_loss.shape)
    
    # print("EIG", eig_val[0].shape)
    C = diag_matrix(eig_val[0], 6)
    # print("矩阵C", C)

    feats_norm = feats_proj / (1+eta*eig_val.unsqueeze(1)) # b,f,eig_dim
    feats_smooth = torch.bmm(feats_norm,eig_vec.permute(0,2,1)) # b,f,n
    return feats_smooth.unsqueeze(3), erci_xing_loss


def spectral_smooth_new1_1(feats, eig_vec, eig_val, eta, lvbo_xishu):
    """
    :param feats:    b,f,n,1
    :param eig_vec:  b,n,eig_dim
    :param eig_val:  b,eig_dim
    :param eta:      float
    :return:         b,f,n,1
    """
    feats_proj = torch.bmm(feats[...,0],eig_vec) # b,f,eig_dim
    # print("eig_val", eig_val.unsqueeze(1).shape)
    # print("feats_proj", feats_proj.shape)
    # lamba_matrix = sum_of_powers((0.5 - eig_val.unsqueeze(1)), 6, lvbo_xishu)
    # print(lamba_matrix)
    
    # print(eig_val)
    # xishu1 = 1 / (1+eta*eig_val.unsqueeze(1))
    # print("Yuan", 1 / (1+eta*eig_val.unsqueeze(1)), xishu1.shape)
    # feats_norm = feats_proj / (1+eta*eig_val.unsqueeze(1)) # b,f,eig_dim
    # print("系数", lvbo_xishu)
    # print(torch.min(lvbo_xishu), torch.max(lvbo_xishu))
    # lvbo_xishu = (lvbo_xishu - torch.min(lvbo_xishu)) / (torch.max(lvbo_xishu) - torch.min(lvbo_xishu))
    # print("系数1", lvbo_xishu)
    # feats_norm = feats_proj * xishu1
    feats_norm = feats_proj * lvbo_xishu
    # print("norm", feats_norm.shape)
    feats_smooth = torch.bmm(feats_norm,eig_vec.permute(0,2,1)) # b,f,n
    
    # 二次型高频系数约束loss
    erci_xing_loss = 0
    if 1:
        feats_proj1 = torch.norm(feats_proj, p=1, dim=(1, 2))  # 'fro'
        feats_proj1 = feats_proj1.unsqueeze(1).unsqueeze(1)
        feats_proj1 = feats_proj / feats_proj1
        # print("fp", feats_proj1)
        A = torch.pow(feats_proj1, 1)
        B = eig_val.unsqueeze(2)
        erci_xing = torch.bmm(A, B)
        erci_xing_loss = torch.norm(erci_xing, p=2, dim=(1, 2))  # p='fro'
        # print("er_loss", erci_xing_loss)
    
    return feats_smooth.unsqueeze(3), erci_xing_loss


def spectral_smooth_yuan(feats, eig_vec, eig_val, eta):
    """
    :param feats:    b,f,n,1
    :param eig_vec:  b,n,eig_dim
    :param eig_val:  b,eig_dim
    :param eta:      float
    :return:         b,f,n,1
    """
    feats_proj = torch.bmm(feats[...,0],eig_vec) # b,f,eig_dim
    feats_norm = feats_proj / (1+eta*eig_val.unsqueeze(1)) # b,f,eig_dim
    feats_smooth = torch.bmm(feats_norm,eig_vec.permute(0,2,1)) # b,f,n
    return feats_smooth.unsqueeze(3)

def compute_smooth_motion_diff(xs,eig_vec,eig_val,eta):
    motion = xs[:, 2:] - xs[:, :2]
    motion_smooth = spectral_smooth_yuan(motion, eig_vec, eig_val, eta)
    motion_diff = motion_smooth - motion
    return motion_diff

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
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

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

def batch_epipolar_distance(x1, x2, F, eps=1e-15):
    '''
    :param x1:  b,n,2
    :param x2:  b,n,2
    :param F:   b,3,3
    :param eps:
    :return: b, n
    '''
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts) # b, n
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = (x2Fx1**2) * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + eps) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + eps))
    return ys
