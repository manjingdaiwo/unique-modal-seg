"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 40   
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-r18-1024-768crop-e40"     #模型名字
weights_path = "model_weights/uavid/{}".format(weights_name)   #模型保存路径
test_weights_name = "last"
log_name = 'uavid/{}'.format(weights_name)   
monitor = 'val_mIoU'   #监控对象
monitor_mode = 'max'   #监控miou的最大值
save_top_k = 3    #保存最好的三个模型
save_last = True   #保存最后一个模型 
check_val_every_n_epoch = 5    #每五轮检查一次验证精度
gpus = [0]    #使用的gpu
strategy = None   #策略
pretrained_ckpt_path = None   #是否预训练
resume_ckpt_path = None        #恢复训练的路径
#  define the network
net = UNetFormer(num_classes=num_classes)    #设置所用的网络模型
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)    #选择损失函数

use_aux_loss = True

# define the dataloader

train_dataset = UAVIDDataset(data_root='data/uavid/train_val', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

val_dataset = UAVIDDataset(data_root='data/uavid/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))



"""打包数据集"""
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

