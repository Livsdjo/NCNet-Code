import argparse
from dataset import Dataset

def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
# 就需要改前两个路径就好
parser.add_argument('--raw_data_path', type=str, default='/home/featurize/data/',
  help='raw data path. default:../raw_data/')
parser.add_argument('--dump_dir', type=str, default='/home/featurize/data/data_dump4/',
  help='data dump path. default:../data_dump')
parser.add_argument('--desc_name', type=str, default='sift-2000',
  help='prefix of desc filename, default:sift-2000')
parser.add_argument('--vis_th', type=int, default=0.35,
  help='visibility threshold')

        
if __name__ == "__main__":
    config = parser.parse_args()
    
    
    test_seqs = ['te-brown1/', 'te-brown2/', 'te-brown3/', 'te-brown4/', 'te-brown5/', 'te-hotel1/', \
                'te-harvard1/', 'te-harvard2/', 'te-harvard3/', 'te-harvard4/', \
                'te-mit1/', 'te-mit2/', 'te-mit3/', 'te-mit4/', 'te-mit5/']
    
    sun3d_te = Dataset(config.raw_data_path+'sun3d_test/', config.dump_dir, 'sun3d-'+config.desc_name+'-test.hdf5', \
        test_seqs, 'test', config.desc_name, \
        config.vis_th, 1000, config.raw_data_path+'pairs/')
    # uncomment these lines if you want generate traning data for SUN3D.
    
    """
        # 已知场景
    # dump yfcc training seqs
    with open('/home/featurize/work/MS2DG-code/MS2DGNET/dump_match/sun3d_train.txt','r') as ofp:
        train_seqs = ofp.read().split('\n')
        del train_seqs[-1]
    train_seqs2 = []
    for seqs in train_seqs:
        seqs_temp = seqs.split('/')
        seqs_temp = seqs_temp[0] + '-' + seqs_temp[1] + '/'
        train_seqs2.append(seqs_temp)
    print(train_seqs2)    
    print('train seq len '+str(len(train_seqs2)))
    yfcc_tr_va = Dataset(config.raw_data_path+'sun3d_train/', config.dump_dir, 'sun3d-'+config.desc_name+'-test-known-123.hdf5', \
        train_seqs2, 'test', config.desc_name, \
        config.vis_th, 100, None)
    """
    
    """
    with open('/home/featurize/work/MS2DG-code/MS2DGNET/dump_match/sun3d_train.txt','r') as ofp:
        train_seqs = ofp.read().split('\n')
    if len(train_seqs[-1]) == 0:
        del train_seqs[-1]
    train_seqs = [seq.replace('/','-')[:-1] for seq in train_seqs]
    print('train seq len '+str(len(train_seqs)))
    """
    """
    sun3d_tr_va = Dataset(config.raw_data_path+'sun3d_train/', config.dump_dir, 'sun3d-'+config.desc_name+'-val.hdf5', \
        train_seqs, 'val', config.desc_name, \
        config.vis_th, 100, None)
    """
    """
    sun3d_tr_tr = Dataset(config.raw_data_path+'sun3d_train/', config.dump_dir, 'sun3d-'+config.desc_name+'-train.hdf5', \
        train_seqs, 'train', config.desc_name, \
        config.vis_th, 2000, None)
    """
