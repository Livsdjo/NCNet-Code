import torch
import numpy as np
import os
import cv2
from six.moves import xrange
from loss import MatchLoss
from evaluation import eval_nondecompose, eval_decompose
from utils import tocuda, get_pool_result
from tqdm import tqdm
from display_file import image_join


def parse_list_file(data_path, list_file):
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


def test_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, _y_gt, config, = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3,3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _mask_before = _valid >= max(0, _valid_th)

    if not config.use_ransac:
        # print("非Ransac 八点发")
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_nondecompose(_x1, _x2, e_hat_out, _dR, _dt, _y_hat_out)
    else:
        # actually not use prob here since probs is None
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_decompose(_x1, _x2, _dR, _dt, mask=_mask_before, method=cv2.RANSAC, \
            probs=None, weighted=False, use_prob=True)
    if _R_hat is None:
        _R_hat = np.random.randn(3,3)
        _t_hat = np.random.randn(3,1)
    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1,-1), _t_hat.reshape(1,-1)]

def dump_res(measure_list, res_path, eval_res, tag):
    """
    # dump test results
    for sub_tag in measure_list:
        # For median error
        ofn = os.path.join(res_path, "median_{}_{}.txt".format(sub_tag, tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(eval_res[sub_tag])))
    """
    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    cur_err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    t_acc_hist, _ = np.histogram(cur_err_t, ths)
    qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    t_acc = np.cumsum(t_acc_hist)
    qt_acc = np.cumsum(qt_acc_hist)
    """
    # Store return val
    for _idx_th in xrange(1, len(ths)):
        ofn = os.path.join(res_path, "acc_q_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
        ofn = os.path.join(res_path, "acc_t_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
        ofn = os.path.join(res_path, "acc_qt_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    ofn = os.path.join(res_path, "all_acc_qt_auc20_{}.txt".format(tag))
    np.savetxt(ofn, np.maximum(cur_err_q, cur_err_t))
    ofn = os.path.join(res_path, "all_acc_q_auc20_{}.txt".format(tag))
    np.savetxt(ofn, cur_err_q)
    ofn = os.path.join(res_path, "all_acc_t_auc20_{}.txt".format(tag))
    np.savetxt(ofn, cur_err_t)
    """
    # Return qt_auc20 
    print("qt_acc", qt_acc)
    ret_val = np.mean(qt_acc[:4])  # 1 == 5
    ret_val2 = np.mean(qt_acc[:3])  # 1 == 5
    ret_val3 = np.mean(qt_acc[:2])  # 1 == 5
    ret_val4 = np.mean(qt_acc[:1])  # 1 == 5
    print(ret_val, ret_val2, ret_val3, ret_val4)
    return ret_val

def denorm(x, T):
    x = (x - np.array([T[0,2], T[1,2]])) / np.asarray([T[0,0], T[1,1]])
    return x

def test_process(mode, model, cur_global_step, data_loader, config):
    model.eval()
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    match_loss = MatchLoss(config)
    loader_iter = iter(data_loader)

    if 0:
        test_seqs = ['buckingham_palace/', 'notre_dame_front_facade/', 'reichstag/', 'sacre_coeur/']
        raw_data_path = '/home/featurize/data/yfcc100m/'
    else:
        test_seqs = ['te-brown1/', 'te-brown2/', 'te-brown3/', 'te-brown4/', 'te-brown5/', 'te-hotel1/', \
                 'te-harvard1/', 'te-harvard2/', 'te-harvard3/', 'te-harvard4/', \
                 'te-mit1/', 'te-mit2/', 'te-mit3/', 'te-mit4/', 'te-mit5/']        
        raw_data_path = '/home/featurize/data/sun3d_test/'

    # save info given by the network
    network_infor_list = ["geo_losses", "cla_losses", "l2_losses", 'precisions', 'recalls', 'f_scores']
    network_info = {info:[] for info in network_infor_list}

    results, pool_arg = [], []
    eval_step, eval_step_i, num_processor = 100, 0, 8
    index = 0
    with torch.no_grad(): 
        for test_data in tqdm(loader_iter):
            test_data = tocuda(test_data)
            res_logits, res_e_hat, loss2, _ , _ = model(test_data)
            y_hat, e_hat = res_logits[-1], res_e_hat[-1]
            loss, geo_loss, cla_loss, prec, rec = match_loss.run(cur_global_step, test_data, y_hat, e_hat)
            info = [geo_loss, cla_loss, prec, rec, 2*prec*rec/(prec+rec+1e-15)]
            # print("结果", geo_loss, cla_loss, prec, rec, 2*prec*rec/(prec+rec+1e-15))
            
            # 显示图像
            if 1 and index > 4000:
            # if 1:
                # [self.seq_index, ii, jj]
                # 1.恢复出图像路径以及原始特征点坐标
                seq_name_index = int(test_data['images'][0][0][0])
                image1_index = int(test_data['images'][0][0][1])
                image2_index = int(test_data['images'][0][0][2])

                seq_name = test_seqs[seq_name_index]

                # dataset_path = config.raw_data_path + seq_name[:-1] + '/test/'
                dataset_path = raw_data_path + seq_name + '/test/'
                image_fullpath_list = parse_list_file(dataset_path, dataset_path + "images.txt")
                image1_path = image_fullpath_list[image1_index]
                image2_path = image_fullpath_list[image2_index]

                # 读取图像
                image1 = cv2.imread(image1_path)
                image2 = cv2.imread(image2_path)
                # 计算label
                gt_geod_d = test_data['ys'][:, :, 0]
                # print(8888, gt_geod_d)
                label_temp = (gt_geod_d < 1e-4).int()
                label_temp = label_temp.cpu().numpy().squeeze(0).astype(np.int16)
                
                logits = y_hat
                mask_temp = (logits > 0).cpu().numpy().squeeze(0).astype(np.int16)
                image = image_join(image1, image2)

                # 画线
                #Setting
                rows0, cols0 = image1.shape[:2]
                rows1, cols1 = image2.shape[:2]
                radius = 4
                thickness = 2
                # is_neg = (gt_geod_d >= 1e-4)
                # print(9999, label_temp, mask_temp)

                xs_4_temp = np.concatenate((test_data['kp_i'][0], test_data['kp_j'][0]), axis=1)
                #画线
               #  for idx in range(min(batch_data['kp_i'][0].shape[0], batch_data['kp_j'][0].shape[0])):
                if 0:
                    for idx, value in enumerate(xs_4_temp):
                        pt0 = (int(value[0]), int(value[1]))
                        pt1 = (int(value[2]) + cols0, int(value[3]))

                        if label_temp[idx] == 1 and mask_temp[idx] == 1:
                            cv2.circle(image, pt0, radius, (0, 255, 0), thickness)
                            cv2.circle(image, pt1, radius, (0, 255, 0), thickness)
                            cv2.line(image, pt0, pt1, (0, 255, 0), thickness)  # color 
                            if idx % 6 == 0:
                                cv2.circle(image, pt0, radius, (0, 0, 255), thickness)
                                cv2.circle(image, pt1, radius, (0, 0, 255), thickness)
                                cv2.line(image, pt0, pt1, (0, 0, 255), thickness)  # color                     
                        elif label_temp[idx] == 1 and mask_temp[idx] == 0:
                            cv2.circle(image, pt0, radius, (0, 0, 255), thickness)
                            cv2.circle(image, pt1, radius, (0, 0, 255), thickness)
                            cv2.line(image, pt0, pt1, (0, 0, 255), thickness)  # color                    
                        else:
                            pass
                else:
                    for idx, value in enumerate(xs_4_temp):
                        pt0 = (int(value[0]), int(value[1]))
                        pt1 = (int(value[2]) + cols0, int(value[3]))

                        if label_temp[idx] == 1:
                            if idx % 1 == 0:
                            #     pass
                            # else:
                                cv2.circle(image, pt0, radius, (0, 255, 0), thickness)
                                cv2.circle(image, pt1, radius, (0, 255, 0), thickness)
                                cv2.line(image, pt0, pt1, (0, 255, 0), thickness)  # color   
                        elif label_temp[idx] == 0:
                            if idx % 1 == 0:
                                pass
                            else:
                                cv2.circle(image, pt0, radius, (0, 0, 255), thickness)
                                cv2.circle(image, pt1, radius, (0, 0, 255), thickness)
                                cv2.line(image, pt0, pt1, (0, 0, 255), thickness)  # color    
                            """
                            if idx % 6 == 0:
                                cv2.circle(image, pt0, radius, (0, 0, 255), thickness)
                                cv2.circle(image, pt1, radius, (0, 0, 255), thickness)
                                cv2.line(image, pt0, pt1, (0, 0, 255), thickness)  # color   
                            """
                        else:
                            pass

                # 显示图像
                # cv2.imshow('Image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite('/home/featurize/images8/output' + str(index)+'.png', image)
            index = index + 1
            
            
            for info_idx, value in enumerate(info):
                network_info[network_infor_list[info_idx]].append(value)

            if config.use_fundamental:
                # unnorm F
                e_hat = torch.matmul(torch.matmul(test_data['T2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['T1s'])
                # get essential matrix from fundamental matrix
                e_hat = torch.matmul(torch.matmul(test_data['K2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['K1s']).reshape(-1,9)
                e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)

            for batch_idx in range(e_hat.shape[0]):
                test_xs = test_data['xs'][batch_idx].detach().cpu().numpy()
                if config.use_fundamental: # back to original
                    x1, x2 = test_xs[0,:,:2], test_xs[0,:,2:4]
                    T1, T2 = test_data['T1s'][batch_idx].cpu().numpy(), test_data['T2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, T1), denorm(x2, T2) # denormalize coordinate
                    K1, K2 = test_data['K1s'][batch_idx].cpu().numpy(), test_data['K2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, K1), denorm(x2, K2) # normalize coordiante with intrinsic
                    test_xs = np.concatenate([x1,x2],axis=-1).reshape(1,-1,4)
                
                pool_arg += [(test_xs, test_data['Rs'][batch_idx].detach().cpu().numpy(), \
                              test_data['ts'][batch_idx].detach().cpu().numpy(), e_hat[batch_idx].detach().cpu().numpy(), \
                              y_hat[batch_idx].detach().cpu().numpy(),  \
                              test_data['ys'][batch_idx,:,0].detach().cpu().numpy(), config)]

                eval_step_i += 1
                if eval_step_i % eval_step == 0:
                    results += get_pool_result(num_processor, test_sample, pool_arg)
                    pool_arg = []
        if len(pool_arg) > 0:
            results += get_pool_result(num_processor, test_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat']
    eval_res = {}
    for measure_idx, measure in enumerate(measure_list):
        eval_res[measure] = np.asarray([result[measure_idx] for result in results])

    if config.res_path == '':
        config.res_path = os.path.join(config.log_path[:-5], mode)
    tag = "ours" if not config.use_ransac else "ours_ransac"
    ret_val = dump_res(measure_list, config.res_path, eval_res, tag)
    
    print("P, R, F", np.mean(np.asarray(network_info['l2_losses'])), np.mean(np.asarray(network_info['precisions'])), \
        np.mean(np.asarray(network_info['recalls'])))
    
    return [ret_val, np.mean(np.asarray(network_info['geo_losses'])), np.mean(np.asarray(network_info['cla_losses'])), \
        np.mean(np.asarray(network_info['l2_losses'])), np.mean(np.asarray(network_info['precisions'])), \
        np.mean(np.asarray(network_info['recalls'])), np.mean(np.asarray(network_info['f_scores']))]


def test(data_loader, model, config):
    # save_file_best = os.path.join(config.model_path, 'model_best.pth')
    # save_file_best = "/home/featurize/log/train/model_best.pth"
    # save_file_best = "/home/featurize/YFCC100M 训练结果log/train/model_best.pth"
    
    # save_file_best = "/home/featurize/SUN3D 训练结果log/train/model_best.pth"
    
    # save_file_best = "/home/featurize/log/train/model_best-75.58.pth"
    # save_file_best = "/home/featurize/YFCC100M 训练结果log/train/model_best-56.4.pth"
    
    # save_file_best = "/home/featurize/YFCC100M_log/train/model_best.pth"
    save_file_best = "/home/featurize/SUN3D 训练结果log/train/model_best-39.4.pth"
    # save_file_best = "/home/featurize/SUN3D_log/train/model_best.pth"
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    else:
        pass
    
    print("进入LMCNet处理")
    checkpoint=torch.load(save_file_best)
    # best_para = checkpoint['best_para']
    # start_epoch = checkpoint['step']
    # model.load_state_dict(checkpoint['network_state_dict'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    
    # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model = torch.nn.DataParallel(model)
    # print(f'==> resuming from step {start_step} best para {best_para}')

    # Restore model
    """
    checkpoint = torch.load(save_file_best)
    start_epoch = checkpoint['epoch']
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    """
    model.cuda()
    print("Restoring from " + str(save_file_best) + ', ' + str(start_epoch) + "epoch...\n")
    if config.res_path == '':
        config.res_path = config.model_path[:-5]+'test'
    print('save result to '+config.res_path)
    va_res = test_process("test", model, 0, data_loader, config)
    print('test result '+str(va_res))
    
def valid(data_loader, model, step, config):
    config.use_ransac = True
    return test_process("valid", model, step, data_loader, config)

