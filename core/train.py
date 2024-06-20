import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda
import torch.distributed as dist

def train_step(step, optimizer, model, match_loss, data, scaler = None):
    model.train()

    # torch.cuda.amp.autocast(True)
    res_logits, res_e_hat, loss2, erci_loss, fitting_loss = model(data) #导入数据
    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        if i == 3:
            # loss_val += [geo_loss, cla_loss, loss2.item(), erci_loss.item(), fitting_loss.item()]
            loss_val += [geo_loss, cla_loss]
        # loss_val += [geo_loss, cla_loss]
    loss += loss2
    optimizer.zero_grad()
    loss.backward()
    # scaler.scale(loss).backward()
    ## scaler.step(optimizer)
    # scaler.update()
    # optimizer.zero_grad()
    
    
    for name, param in model.named_parameters():
         # print("param", param)
         # if param.grad is None:
         #   print(name, param.grad)
        
         
         if torch.any(torch.isnan(param.grad)):
            # print(torch.isnan(param.grad))
            # print('skip because nan')
            return loss_val
        
    
    optimizer.step()
    return loss_val


def train(model, train_loader, valid_loader, config):
    model.cuda()
    """dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:21002',
        world_size=1,
        rank=0
    )
    model = torch.nn.parallel.DistributedDataParallel(model)"""
    # model = torch.nn.DataParallel(model, device_ids=[0,1])
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config)
    # scaler = torch.cuda.amp.GradScaler()

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    
    """
        checkpoint=torch.load(self.pth_fn)
        best_para = checkpoint['best_para']
        start_step = checkpoint['step']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'==> resuming from step {start_step} best para {best_para}')
    """
    # if config.resume:
    if 0:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan') #主要代码的文件名
        logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss']*(config.iter_num+2))
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])
    train_loader_iter = iter(train_loader[0])
    train_loader_iter1 = iter(train_loader[1])
    flag = 0
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            if step % 2 == 0:
                flag = 1
                train_data = next(train_loader_iter)
            else:
                flag = 2
                train_data = next(train_loader_iter1)
        except StopIteration:
            if flag == 1:
                train_loader_iter = iter(train_loader[0])
                train_data = next(train_loader_iter)
            elif flag == 2:
                train_loader_iter1 = iter(train_loader[1])
                train_data = next(train_loader_iter1)  
            else:
                pass
        train_data = tocuda(train_data)
        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals = train_step(step, optimizer, model, match_loss, train_data, None)   #训练
        # print("vals", loss_vals)
        logger_train.append([cur_lr] + loss_vals)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  P, R, F  = valid(valid_loader, model, step, config)   # 验证
            logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))
        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

