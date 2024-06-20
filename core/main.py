# Code is heavily borrowed from https://github.com/vcg-uvic/learned-correspondence-release
# Author: Jiahui Zhang
# Date: 2019/09/03
# E-mail: jiahui-z15@mails.tsinghua.edu.cn


from config import get_config, print_usage
config, unparsed = get_config()
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
# from ms2dgnet import MS2DNET as Model
# from network.lmcnet2 import LMCNet as Model
# 自己的网络
from network.OurNet import LMCNet as Model

# from network.OurNet_No_GCN import LMCNet as Model
# from network.OurNet_No_Feature import LMCNet as Model

from train import train
from test import test
import torch
from thop import profile
import yaml

# from utils.base_utils import load_cfg

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")

def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    #result_path = config.log_base+'/'+suffix
    result_path = './log'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path+'/train'):
        os.makedirs(result_path+'/train')
    if not os.path.isdir(result_path+'/valid'):
        os.makedirs(result_path+'/valid')
    if not os.path.isdir(result_path+'/test'):
        os.makedirs(result_path+'/test')
    if os.path.exists(result_path+'/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path+'/config.th')

    # path for saving traning logs
    config.log_path = result_path+'/train'

def main(config):
    use_network = "LMCNet"
    
    """The main function."""
    # Initialize network
    if use_network == "ms2dgnet":
        model = Model(config)
        model = model.cuda()
    elif use_network == "LMCNet":
        config1 = load_cfg(config.network_cfg)
        model = Model(config1)
        model = model.cuda()
    else:
        print("错误.......")


    # Run propper mode
    if config.run_mode == "train":
        create_log_dir(config)

        train_dataset = CorrespondencesDataset(config.data_tr, config)
        train_dataset1 = CorrespondencesDataset("/home/featurize/data/yfcc-sift-2000-train1.hdf5", config)
        # train_dataset1 = CorrespondencesDataset("/home/featurize/data/sun3d-sift-2000-train.hdf5", config)

        
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=32, shuffle=True,
                num_workers=32, pin_memory=True, collate_fn=collate_fn)
        
        train_loader1 = torch.utils.data.DataLoader(
                train_dataset1, batch_size=32, shuffle=True,
                num_workers=32, pin_memory=True, collate_fn=collate_fn)
        """
        valid_dataset = CorrespondencesDataset(config.data_va, config)
        valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=8, shuffle=False,
                num_workers=8, pin_memory=True, collate_fn=collate_fn)
        """
        test_dataset = CorrespondencesDataset(config.data_te, config)
        valid_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False,
                num_workers=8, pin_memory=False, collate_fn=collate_fn)
        #valid_loader = None
        print('start training .....')
        train(model, [train_loader, train_loader1], valid_loader, config)#在训练之前要把(model, train_loader, valid_loader, config)准备好

    elif config.run_mode == "test":
        test_dataset = CorrespondencesDataset(config.data_te, config)
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False,
                num_workers=8, pin_memory=False, collate_fn=collate_fn)

        test(test_loader, model, config)




if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

#
# main.py ends here
