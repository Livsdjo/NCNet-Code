import torch
import numpy as np
import os
import cv2
from six.moves import xrange
from loss import MatchLoss
from evaluation import eval_nondecompose, eval_decompose
from utils import tocuda, get_pool_result
import h5py



if __name__ == "__main__":
    data = h5py.File('/home/featurize/work/data_dump2/yfcc-sift-2000-test-LMCNet.hdf5','r')
    print(data.keys())
    
    