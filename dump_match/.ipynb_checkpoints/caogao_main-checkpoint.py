from __future__ import print_function
import numpy as np
import sys
from tqdm import tqdm
import os
import pickle
import cv2
import itertools
from six.moves import xrange
from feature_match import computeNN
from utils import saveh5, loadh5
from geom import load_geom, parse_geom, get_episym
from transformations import quaternion_from_matrix
from eig_utils import build_graph, name2config

class Sequence(object):
    def __init__(self, dataset_path, dump_dir, desc_name, vis_th, pair_num, seq_index, pair_name=None):
        self.data_path = dataset_path.rstrip("/") + "/"
        self.dump_dir = dump_dir
        self.desc_name = desc_name
        self.seq_index = seq_index
        self.eig_name = "small_min"
        print('dump dir ' + self.dump_dir)
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)
        self.intermediate_dir = os.path.join(self.dump_dir, 'dump')
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
        img_list_file = self.data_path + "images.txt"
        geom_list_file = self.data_path + "calibration.txt"
        vis_list_file = self.data_path + "visibility.txt"
        self.image_fullpath_list = self.parse_list_file(self.data_path, img_list_file)
        self.geom_fullpath_list = self.parse_list_file(self.data_path, geom_list_file)
        self.vis_fullpath_list = self.parse_list_file(self.data_path, vis_list_file)
        # load geom and vis
        self.geom, self.vis = [], []
        for geom_file, vis_file in zip(self.geom_fullpath_list, self.vis_fullpath_list):
            self.geom += [load_geom(geom_file)]
            self.vis += [np.loadtxt(vis_file).flatten().astype("float32")]
        self.vis = np.asarray(self.vis)
        img_num = len(self.image_fullpath_list)
        if pair_name is None:
            self.pairs = []
            for ii, jj in itertools.product(xrange(img_num), xrange(img_num)):
                if ii != jj and self.vis[ii][jj] > vis_th:
                    self.pairs.append((ii, jj))
            # 每次生成都是一样的
            np.random.seed(1234)
            self.pairs = [self.pairs[i] for i in np.random.permutation(len(self.pairs))[:pair_num]]

            # 写入
            # name = pair_name
            # with open(pair_name, 'wb') as f:
            #     pickle.dump(self.pairs, f)
        else:
            with open(pair_name, 'rb') as f:
                self.pairs = pickle.load(f)
        print('pair lens' + str(len(self.pairs)))

    def dump_nn(self, ii, jj):
        dump_file = os.path.join(self.intermediate_dir, "nn-{}-{}.h5".format(ii, jj))
        if not os.path.exists(dump_file):
            # os.makedirs(dump_file)
            image_i, image_j = self.image_fullpath_list[ii], self.image_fullpath_list[jj]
            desc_ii = loadh5(image_i+'.'+self.desc_name+'.hdf5')["descriptors"]
            desc_jj = loadh5(image_j+'.'+self.desc_name+'.hdf5')["descriptors"]
            idx_sort, ratio_test, mutual_nearest = computeNN(desc_ii, desc_jj)
            # Dump to disk
            dump_dict = {}
            dump_dict["idx_sort"] = idx_sort
            dump_dict["ratio_test"] = ratio_test
            dump_dict["mutual_nearest"] = mutual_nearest
            saveh5(dump_dict, dump_file)

    def dump_intermediate(self):
        for ii, jj in tqdm(self.pairs):
            self.dump_nn(ii, jj)
        print('Done')

    def unpack_K(self, geom):
        img_size, K = geom['img_size'], geom['K']
        w, h = img_size[0], img_size[1]
        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5
        cx += K[0, 2]
        cy += K[1, 2]
        # Get focals
        fx = K[0, 0]
        fy = K[1, 1]
        return cx,cy,[fx,fy]

    def norm_kp(self, cx, cy, fx, fy, kp):
        # New kp
        kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
        return kp

    def make_xy(self, ii, jj):
        geom_i, geom_j = parse_geom(self.geom[ii]), parse_geom(self.geom[jj])
        # should check the image size here
        #load img and check img_size
        image_i, image_j = self.image_fullpath_list[ii], self.image_fullpath_list[jj]
        kp_i = loadh5(image_i+'.'+self.desc_name+'.hdf5')["keypoints"][:, :2]
        kp_j = loadh5(image_j+'.'+self.desc_name+'.hdf5')["keypoints"][:, :2]
        cx1, cy1, f1 = self.unpack_K(geom_i)
        cx2, cy2, f2 = self.unpack_K(geom_j)
        ## 第一次归一化
        x1 = self.norm_kp(cx1, cy1, f1[0], f1[1], kp_i)
        x2 = self.norm_kp(cx2, cy2, f2[0], f2[1], kp_j)
        R_i, R_j = geom_i["R"], geom_j["R"]
        dR = np.dot(R_j, R_i.T)
        t_i, t_j = geom_i["t"].reshape([3, 1]), geom_j["t"].reshape([3, 1])
        dt = t_j - np.dot(dR, t_i)
        if np.sqrt(np.sum(dt**2)) <= 1e-5:
            return []
        dtnorm = np.sqrt(np.sum(dt**2))
        dt /= dtnorm
        nn_info = loadh5(os.path.join(self.intermediate_dir, "nn-{}-{}.h5".format(ii, jj)))
        idx_sort, ratio_test, mutual_nearest = nn_info["idx_sort"], nn_info["ratio_test"], nn_info["mutual_nearest"]
        x2 = x2[idx_sort[1],:]
        kp_j = kp_j[idx_sort[1],:]
        xs = np.concatenate([x1, x2], axis=1).reshape(1,-1,4)
        geod_d = get_episym(x1, x2, dR, dt)
        ys = geod_d.reshape(-1,1)

        # print(f1, cx1)
        K1 = np.asarray([
            [f1[0], 0, cx1],
            [0, f1[1], cy1],
            [0, 0, 1]
        ])
        K2 = np.asarray([
            [f2[0], 0, cx2],
            [0, f2[1], cy2],
            [0, 0, 1]
        ])

        x3 = self.hpts_to_pts(self.pts_to_hpts(kp_i) @ np.linalg.inv(K1).T)
        x4 = self.hpts_to_pts(self.pts_to_hpts(kp_j) @ np.linalg.inv(K2).T)
        x5 = np.concatenate([x3, x4], 1)

        # print("计算特征值", x5.shape, x5)
        eig_val, eig_vec = build_graph(x5, name2config[self.eig_name])
        # print("66666", eig_val.shape, eig_vec.shape)
        eig_data = np.concatenate([eig_val[None, :], eig_vec], 0).astype(np.float32)
        # print("eig_data", eig_data)

        return xs, ys, dR, dt, ratio_test, mutual_nearest, cx1, cy1, f1, cx2, cy2, f2, kp_i, kp_j, [self.seq_index, ii, jj], eig_data

    def pts_to_hpts(self, pts):
        return np.concatenate([pts, np.ones([pts.shape[0], 1])], 1)

    def hpts_to_pts(self, hpts):
        return hpts[:, :2] / hpts[:, 2:]

    def dump_datasets(self):
        ready_file = os.path.join(self.dump_dir, "ready")
        var_name = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals', 'cx1s', 'cy1s', 'f1s', 'cx2s', 'cy2s', 'f2s', 'kp_i', 'kp_j', 'images', 'eig_data']
        res_dict = {}
        for name in var_name:
            res_dict[name] = []
        if not os.path.exists(ready_file):
            print("\n -- No ready file {}".format(ready_file))
            for pair_idx, pair in enumerate(self.pairs):
                print("\rWorking on {} / {}".format(pair_idx, len(self.pairs)), end="")
                sys.stdout.flush()
                res = self.make_xy(pair[0], pair[1])
                if len(res)!=0:
                    for var_idx, name in enumerate(var_name):
                        res_dict[name] += [res[var_idx]]
            # 一个Sequence所有的ii和jj
            for name in var_name:
                out_file_name = os.path.join(self.dump_dir, name) + ".pkl"
                print(333, out_file_name)
                with open(out_file_name, "wb") as ofp:
                    pickle.dump(res_dict[name], ofp)
            """
            # Mark ready
            with open(ready_file, "w") as ofp:
                ofp.write("This folder is ready\n")
            """
        else:
             print('Done!')   


    def parse_list_file(self, data_path, list_file):
        fullpath_list = []
        with open(list_file, "r") as img_list:
            while True:
                # read a single line
                tmp = img_list.readline()
                if type(tmp) != str:
                    line2parse = tmp.decode("utf-8")
                else:
                    line2parse = tmp
                if not line2parse:
                    break
                # strip the newline at the end and add to list with full path
                fullpath_list += [data_path + line2parse.rstrip("\n")]
        return fullpath_list

    
    def collect(self):
        data_type = ['xs','ys','Rs','ts', 'ratios', 'mutuals',\
            'cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s', 'kp_i', 'kp_j', 'images', 'eig_data']
        # data_type = ['xs','ys','Rs','ts', 'ratios', 'mutuals',\
        #    'cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s', 'kp_i', 'kp_j', 'images']
        pair_idx = 0
        with h5py.File(self.dump_file, 'w') as f:
            data = {}
            for tp in data_type:
                data[tp] = f.create_group(tp)
            for seq in self.seqs:
                print(seq)
                data_seq = {}
                for tp in data_type:
                    print(1111, self.dump_dir+'/'+seq+'/'+self.desc_name+'/'+self.mode+'/'+str(tp)+'.pkl')
                    data_seq[tp] = pickle.load(open(self.dump_dir+'/'+seq+'/'+self.desc_name+'/'+self.mode+'/'+str(tp)+'.pkl','rb'))
                seq_len = len(data_seq['xs'])

                for i in range(seq_len):
                    for tp in data_type:
                        data_item = data_seq[tp][i]
                        if tp in ['cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s', 'images']:
                            data_item = np.asarray([data_item])
                        data_i = data[tp].create_dataset(str(pair_idx), data_item.shape, dtype=np.float32)
                        data_i[:] = data_item.astype(np.float32)
                    pair_idx = pair_idx + 1
                print('pair idx now ' +str(pair_idx))

    
    
    
if __name__ == '__main__':
    """
    with open("/home/featurize/data/data_dump6/big_ben_1/sift-2000/train/images.pkl", "r") as ofp:
        data = ofp.read()
        print(data)
    """
    data = pickle.load(open("/home/featurize/data/data_dump6/big_ben_1/sift-2000/train/images.pkl",'rb'))
    data1 = pickle.load(open("/home/featurize/data/data_dump6/big_ben_1/sift-2000/train/xs.pkl",'rb'))
    print(len(data1), len(data))
    
    
    data_type = ['xs','ys','Rs','ts', 'ratios', 'mutuals',\
            'cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s', 'images', 'eig_data']
    with open('/home/featurize/work/con_e/yfcc_train.txt','r') as ofp:
        test_seqs = ofp.read().split('\n')
    if len(test_seqs[-1]) == 0:
        del test_seqs[-1]
    print("seqs", test_seqs)
    print(test_seqs)
    for seq in tqdm(test_seqs):
        if seq == "piazza_dei_miracoli":
            break
        """
        if seq == "blue_mosque_interior_1":
            pass
        else:
            continue
        """
        for type1 in data_type:            
           path = "/home/featurize/data/data_dump6/" + seq + "/sift-2000/train/" + type1 + ".pkl"
           data = pickle.load(open(path,'rb'))
           if len(data) >5000:
              data1 = data[:5000]
           else:
              data1 = data
           # print(data1)
        
           with open(path, "wb") as ofp:
               pickle.dump(data1, ofp)
    
    
        