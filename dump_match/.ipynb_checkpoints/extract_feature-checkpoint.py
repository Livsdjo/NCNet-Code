import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2
import h5py


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, 
  # default='/home/featurize/data/raw_data/yfcc100m/',
  default='/home/featurize/data/raw_data/sun3d_train/',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--img_glob', type=str, 
  default='*/test/images/*.jpg',
  # default='*/*/images/*.jpg',
  # default='*/images/*.jpg',
  help='Glob match if directory of images is specified (default: \'*/images/*.jpg\').')
parser.add_argument('--num_kp', type=int, default='2000',
  help='keypoint number, default:2000')
parser.add_argument('--suffix', type=str, default='sift-2000',
  help='suffix of filename, default:sift-2000')



class ExtractSIFT(object):
  def __init__(self, num_kp, contrastThreshold=1e-5):
    # self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
    self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    
  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.sift.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

def write_feature(pts, desc, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors"][:] = desc

if __name__ == "__main__":
  opt = parser.parse_args()
  detector = ExtractSIFT(opt.num_kp)

  # ---------------------------------第一版--------------------------
  # get image lists
  # search = os.path.join(opt.input_path, opt.img_glob)
  # listing = glob.glob(search)

  # test_seqs = ['buckingham_palace', 'notre_dame_front_facade', 'reichstag', 'sacre_coeur']
  # test_seqs = ['te-brown1/', 'te-brown2/', 'te-brown3/', 'te-brown4/', 'te-brown5/', 'te-hotel1/', \
  #              'te-harvard1/', 'te-harvard2/', 'te-harvard3/', 'te-harvard4/', \
  #              'te-mit1/', 'te-mit2/', 'te-mit3/', 'te-mit4/', 'te-mit5/']
  """
  for test_seq_name in test_seqs:
      # get image lists
      # search = os.path.join(opt.input_path, opt.img_glob)
      input_data_1 = opt.input_path + test_seq_name + '/'
      search = os.path.join(input_data_1, opt.img_glob)
      print("se", search)
      listing = glob.glob(search)
      

      for img_path in tqdm(listing):
        kp, desc = detector.run(img_path)
        # 描述子 特征点
        save_path = img_path+'.'+opt.suffix+'.hdf5'
        write_feature(kp, desc, save_path)
  """
  # ---------------------------------------------------------------------------------------
  # get image lists
  search = os.path.join(opt.input_path, opt.img_glob)
  listing = glob.glob(search)

  for img_path in tqdm(listing):
    kp, desc = detector.run(img_path)
    # 描述子 特征点
    save_path = img_path+'.'+opt.suffix+'.hdf5'
    write_feature(kp, desc, save_path)
  



