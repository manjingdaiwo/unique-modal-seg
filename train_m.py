import os
from xmlrpc.client import boolean

import time
import torch
import argparse
import torch.distributed as dist
from catalyst.contrib.nn import Lookahead
import random
import numpy as np
from torch.backends import cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import torch.autograd
from skimage import io
import pandas as pd
from collections import OrderedDict
from datetime import datetime
#multi-gpu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.utils import AverageMeter
from tools.cfg import py2cfg
from tools.metric import Evaluator

def get_args():
    """配置接口"""
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-r', "--resume", action = 'store_true', help='whether resume train from a saved model')
    arg('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')# multigpu
    return parser.parse_args()

#获取最新文件
def get_newest_file(path_file):
    path_file = str(path_file)
    lists = os.listdir(path_file)
    lists.sort(key=lambda fn: os.path.getmtime(os.path.join(path_file, fn)))
    file_newest = os.path.join(path_file,lists[-1])
    return file_newest

#及时清理文件
def remove_older_file(path_file):
    path_file = str(path_file)
    lists = os.listdir(path_file)
    lists.sort(key=lambda fn: os.path.getmtime(path_file +'/'+fn))
    if len(lists) >= 6:
        if os.path.exists(os.path.join(path_file, lists[0])):
            os.remove(os.path.join(path_file, lists[0]))

#随机种子
def seed_everything(seed):
    """随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

args = get_args()
config = py2cfg('./config/loveda/unetformer.py')
metrics_train = Evaluator(num_class=config.num_classes)
metrics_val = Evaluator(num_class=config.num_classes)

if not args.resume:
    #在重载模型时，不创建新的文件
    if not os.path.exists(config.chkpt_dir_best_m): 
        os.makedirs(config.chkpt_dir_best_m)

    if not os.path.exists(config.chkpt_dir_regular_m): 
        os.makedirs(config.chkpt_dir_regular_m)

    if not os.path.exists(config.log_dir_m): 
        os.makedirs(config.log_dir_m) 
#日志记录
df = pd.DataFrame(columns=['time', 'epoch', 'lr','train_loss', 'Train_OA', 'Val_OA'])
# df.to_csv(config.log_dir_m+'/train_log.csv',mode='a',index=False) 

#模型的重载
if args.resume:
    pth = get_newest_file("/home/guest/Pycode/3/GeoSeg-main/checkpoints_m/Mynet1/LoveDA/best/")
    state_dict = torch.load(pth)
    state_dict_model = state_dict['model_state_dict']
    # checkpoint_op = state_dict['optimizer_state_dict']
    # checkpoint_lr = state_dict['lr']
    # checkpoint_ep = state_dict['epoch']
    # checkpoint_acc = state_dict['bestaccV']

    checkpoint = OrderedDict()

    for k, v in state_dict_model.items():
        name = k[7:]  # remove `module.`
        checkpoint[name] = v

def main():
    dist.init_process_group(backend='nccl') # multigpu
    torch.cuda.set_device(args.local_rank)# multigpu

    seed_everything(12345)
    net = config.net
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=config.lr * 0.000001, T_0=15, T_mult=2)
    
    if config.gpu:
        net = net.cuda()
    if args.resume:
        # optimizer.load_state_dict(6)
        # lr_sch.load_state_dict(config.lr)
        print(pth+' resumed successfully.')

    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], find_unused_parameters=True)#multi gpu

    train_dataset = config.train_dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) # multigpu
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, sampler=train_sampler,pin_memory=True,drop_last=True) # multigpu

    val_loader = config.val_loader
    criterion = config.loss.cuda()

    train(train_loader, net, criterion, optimizer, val_loader, lr_sch)
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, val_loader, lr_sch):
    bestaccT=0
    bestloss=1
    if args.resume:
        curr_epoch = 6 + 1
        bestaccV= 0.5
        best_epoch = 6
    else:
        curr_epoch = 1
        bestaccV=0.0001
        best_epoch = 1
    begin_time = time.time()

    while True:
        torch.cuda.empty_cache()
        net.train()       
        train_loss = AverageMeter()
        loop = tqdm(enumerate(train_loader), total =len(train_loader), colour = 'GREEN')

        for i, data in loop:
            imgs, labels = data['img'], data['gt_semantic_seg']
            if config.gpu:
                 imgs = imgs.cuda(non_blocking=True).float()
                 labels = labels.cuda(non_blocking=True).long()
            # print("\ntrain_dataset", imgs.size())
            outputs = net(imgs)     
            main_loss = criterion(outputs, labels)
            loss = main_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lr_sch.step()

            labels = labels.cpu().detach().numpy()
            preds = nn.Softmax(dim=1)(outputs)
            preds = preds.cpu().detach()
            pre_label = preds.argmax(dim=1)
            pre_label = pre_label.numpy()

            for i in range(labels.shape[0]):
                metrics_train.add_batch(labels[i], pre_label[i])
                OA = np.nanmean(metrics_train.OA())
            train_loss.update(loss.cpu().detach().numpy())

            lr = optimizer.param_groups[0]['lr']
            loop.set_description(f'Epoch [{curr_epoch}/{config.max_epoch}]')
            loop.set_postfix(loss = np.round(train_loss.val, 4), OA = np.round(OA, 4), lr = lr)

        if config.DATA_NAME == 'LoveDA':           
            mIoU = np.nanmean(metrics_train.Intersection_over_Union())
            F1 = np.nanmean(metrics_train.F1())
        else:
            mIoU = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(metrics_train.F1()[:-1])
        OA = np.nanmean(metrics_train.OA())
        eval_value = {'mIoU': np.round(mIoU, 4), 'F1': np.round(F1, 4), 'OA': np.round(OA, 4)}
        print('Train:', eval_value)
        metrics_train.reset()
        acc_OA, loss_v = validate(val_loader, net, criterion)
        if dist.get_rank() == 0:
            list = ["%s"%datetime.now(), curr_epoch, lr, np.round(train_loss.val, 4), np.round(OA, 4), np.round(acc_OA, 4)]
            data = pd.DataFrame([list])
            data.to_csv(config.log_dir_m+'/train_log.csv',mode='a',header=False,index=False)  #mode:a,追加数据

        #保存目前最好模型
        if acc_OA>=bestaccV:
            bestaccV=acc_OA
            bestloss=loss_v
            if dist.get_rank() == 0:
                save_path = os.path.join(config.chkpt_dir_best_m, config.NET_NAME+'_%de_OA%.2f_best.pth'%(curr_epoch, acc_OA*100))
                torch.save({
                    'model_state_dict': net.state_dict(),
                    }, save_path)
                print("###saved best model to the {}".format(save_path))
                remove_older_file(config.chkpt_dir_best_m)
            best_epoch = curr_epoch
            bestaccT = OA 
            
        #每五个epoch保存一次模型
        if curr_epoch % config.save_per_epoch == 0:
            if dist.get_rank() == 0:
                save_path = os.path.join(config.chkpt_dir_regular_m, config.NET_NAME+'_%de_OA%.2f_regular.pth'%(curr_epoch, acc_OA*100))
                torch.save({
                    'model_state_dict': net.state_dict(),
                    }, save_path)
                remove_older_file(config.chkpt_dir_regular_m)
                print("###saved regular modelto the {}".format(save_path))

        time_s = (time.time() - begin_time) / 3600
        print('Total time: %.3f hour,  Best rec: %dth epoch : Train %.2f, Val %.2f, Val_loss %.4f\n'
              %(time_s, best_epoch, bestaccT*100, bestaccV*100, bestloss))
        curr_epoch += 1
        if curr_epoch >= config.max_epoch+1:
            return

def validate(val_loader, net, criterion):
    net.eval()         
    torch.cuda.empty_cache()
    val_loss = AverageMeter()
    with torch.no_grad():
        for _, data in enumerate(val_loader):

            imgs, labels = data['img'], data['gt_semantic_seg']

            if config.gpu:
                imgs = imgs.cuda().float()
                labels = labels.cuda().long()

            outputs = net(imgs)
            loss = criterion(outputs, labels)

            val_loss.update(loss.cpu().detach().numpy()) 

            labels = labels.cpu().detach().numpy()
            preds = nn.Softmax(dim=1)(outputs)
            preds = preds.cpu().detach()
            pre_label = preds.argmax(dim=1)
            pre_label = pre_label.numpy()

            for i in range(labels.shape[0]):
                metrics_val.add_batch(labels[i], pre_label[i])

        if config.DATA_NAME == 'LoveDA':           
            mIoU = np.nanmean(metrics_val.Intersection_over_Union())
            F1 = np.nanmean(metrics_val.F1())
        else:
            mIoU = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(metrics_val.F1()[:-1])
        OA = np.nanmean(metrics_val.OA())
        iou_per_class = metrics_val.Intersection_over_Union()
        eval_value = {'mIoU': np.round(mIoU, 4), 'F1': np.round(F1, 4), 'OA': np.round(OA, 4)}
        print('Val:', eval_value)

        iou_value = {}
        for class_name, iou in zip(config.classes, iou_per_class):
            iou_value[class_name] = np.round(iou, 4)
        print("val per_class iou", iou_value)
        metrics_val.reset()

        return OA, val_loss.avg
        
if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    main()
    
 # CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 train_m.py
