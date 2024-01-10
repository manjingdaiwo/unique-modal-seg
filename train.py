import os
from xmlrpc.client import boolean
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import torch
import argparse
import torch.distributed as dist

import random
import numpy as np
from torch.backends import cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import torch.autograd
from skimage import io
import pandas as pd

from tools.utils import AverageMeter
from tools.cfg import py2cfg
from tools.metric import Evaluator

from torch.backends import cudnn
cudnn.benchmark = True

"""配置接口"""
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg('-r', "--resume", action = 'store_true', help='whether resume train from a saved model')
    return parser.parse_args()

#获取最新文件
def get_newest_file(path_file):
    path_file = str(path_file)
    lists = os.listdir(path_file)
    lists.sort(key=lambda fn: os.path.getmtime(os.path.join(path_file, fn)))
    file_newest = os.path.join(path_file,lists[-2])
    return file_newest

#及时清理文件
def remove_older_file(path_file):
    path_file = str(path_file)
    lists = os.listdir(path_file)
    lists.sort(key=lambda fn: os.path.getmtime(path_file +'/'+fn))
    if len(lists) >= 6:
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
config = py2cfg(args.config_path)
metrics_train = Evaluator(num_class=config.num_classes)
metrics_val = Evaluator(num_class=config.num_classes)

#在重载模型时，不创建新的文件
if not args.resume:
    f = open('last_dir.txt', 'a+')
    f.truncate(0)
    if not os.path.exists(config.chkpt_dir_best): 
        os.makedirs(config.chkpt_dir_best)
        f.write('{}\n'.format(config.chkpt_dir_best))

    if not os.path.exists(config.chkpt_dir_regular): 
        os.makedirs(config.chkpt_dir_regular)
        f.write('{}\n'.format(config.chkpt_dir_regular))

    if not os.path.exists(config.log_dir): 
        os.makedirs(config.log_dir)
        f.write(config.log_dir)
    f.close()
#获取上一次的训练模型路径,在重载时方便管理文件
dir_list = np.loadtxt('last_dir.txt', dtype=str)   
#日志记录
df = pd.DataFrame(columns=['time', 'epoch', 'lr','train_loss', 'Train_OA', 'Val_OA'])
df.to_csv(dir_list[2]+'/train_log.csv',mode='a',index=False) 

#模型的重载
if args.resume:
    pth = get_newest_file(dir_list[0])
    checkpoint = torch.load(pth)

def main():
    seed_everything(43)
    net = config.net
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=config.lr * 0.0001, T_0=15, T_mult=2)
    
    if config.gpu:
        net = net.cuda()
        # print(net)
    if args.resume:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_sch.load_state_dict(checkpoint['lr'])

    train_loader = config.train_loader

    val_loader = config.val_loader
    criterion = config.loss.cuda()

    train(train_loader, net, criterion, optimizer, val_loader, lr_sch)
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, val_loader, lr_sch):
    bestaccT=0
    bestaccV=0
    bestloss=1
    if args.resume:
        curr_epoch = checkpoint['epoch'] + 1
    else:
        curr_epoch = 1
    best_epoch = 1
    begin_time = time.time()

    while True:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        net.train()       
        train_loss = AverageMeter()
        loop = tqdm(enumerate(train_loader), total =len(train_loader), colour = 'GREEN')

        repetitions = len(train_loader.dataset)
        timings = np.zeros((repetitions, 1))

        for i, data in loop:
            imgs, labels = data['img'], data['gt_semantic_seg']
            if config.gpu:
                 imgs = imgs.cuda().float()
                 labels = labels.cuda().long()
            starter.record()
            outputs = net(imgs) 
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成 
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[i] = curr_time   
            main_loss = criterion(outputs, labels, mode='train')
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
                    
        mIoU = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
        F1 = np.nanmean(metrics_train.F1()[:-1])
        OA = np.nanmean(metrics_train.OA())
        eval_value = {'mIoU': np.round(mIoU, 4), 'F1': np.round(F1, 4), 'OA': np.round(OA, 4)}
        print('Train:', eval_value)
        metrics_train.reset()

        acc_OA, loss_v = validate(val_loader, net, criterion)

        list = [config.TIME_NOW, curr_epoch, lr, np.round(train_loss.val, 4), np.round(OA, 4), np.round(acc_OA, 4)]
        data = pd.DataFrame([list])
        data.to_csv(dir_list[2]+'/train_log.csv',mode='a',header=False,index=False)  #mode:a,追加数据


        #保存目前最好模型
        if acc_OA>=bestaccV:
            bestaccV=acc_OA
            bestloss=loss_v
            save_path = os.path.join(dir_list[0], config.NET_NAME+'_%de_OA%.2f_best.pth'%(curr_epoch, acc_OA*100))
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'epoch':curr_epoch,
                'lr':lr_sch.state_dict(),
                }, save_path)
            best_epoch = curr_epoch
            bestaccT = OA
            remove_older_file(dir_list[0])
            print("###saved best model to the {}".format(save_path))
            
        #每五个epoch保存一次模型
        if curr_epoch % config.save_per_epoch == 0 :
            save_path = os.path.join(dir_list[1], config.NET_NAME+'_%de_OA%.2f_regular.pth'%(curr_epoch, acc_OA*100))
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'epoch':curr_epoch,
                'lr':lr_sch.state_dict(),
                }, save_path)
            remove_older_file(dir_list[0])
            print("###saved regular modelto the {}".format(save_path))

        time_s = (time.time() - begin_time) / 3600
        print('Total time: %.3f hour,  Best rec: %dth epoch : Train %.2f, Val %.2f, Val_loss %.4f\n'
              %(time_s, best_epoch, bestaccT*100, bestaccV*100, bestloss))
        curr_epoch += 1
        if curr_epoch >= config.max_epoch+1:
            avg = timings.sum()/(repetitions*1000)
            print('\navg time={}s\n'.format(avg))
            return
def validate(val_loader, net, criterion):
    net.eval()         
    torch.cuda.empty_cache()
    val_loss = AverageMeter()

    for _, data in enumerate(val_loader):

        imgs, labels = data['img'], data['gt_semantic_seg']

        if config.gpu:
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()

        with torch.no_grad():
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
    main()
