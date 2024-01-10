from torch.utils.data import DataLoader
from geoseg.losses import *
from models.lanet import Mynet as Net
from datetime import datetime
from IPython import embed
from geoseg.datasets.vaihingen_dataset import *
# training hparam
max_epoch = 200
ignore_index = len(CLASSES)

train_batch_size = 16
val_batch_size = 4
test_batch_size = 1

lr = 6e-4
weight_decay = 0.01

accumulate_n = 4
num_classes = len(CLASSES)
classes = CLASSES

save_per_epoch = 5

gpu = True
gpus = [0]

pretrained_ckpt_path = None

# net = UTNet(3, 32, num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True).cuda()
net = Net(3, num_classes).cuda()
NET_NAME = 'lanet'
DATA_NAME = 'vai'
working_path = '/home/a1/ss/semantic_segmentation/my_segmodel'

# DATE_FORMAT = '%d_%B_%Hh_%Mm'
DATE_FORMAT = '%m-%d-%H:%M'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

chkpt_dir_best = os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, TIME_NOW, 'best')
chkpt_dir_regular = os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, TIME_NOW, 'regular')
log_dir = os.path.join(working_path, 'logs', NET_NAME, DATA_NAME,  TIME_NOW)

chkpt_dir_best_m = os.path.join(working_path, 'checkpoints_m', NET_NAME, DATA_NAME, TIME_NOW, 'best')
chkpt_dir_regular_m = os.path.join(working_path, 'checkpoints_m', NET_NAME, DATA_NAME, TIME_NOW, 'regular')
log_dir_m = os.path.join(working_path, 'logs_m', NET_NAME, DATA_NAME, TIME_NOW)

pred_dir_preds = os.path.join(working_path, 'results',  NET_NAME, DATA_NAME, 'preds')
pred_dir_labels = os.path.join(working_path, 'results',  NET_NAME, DATA_NAME, 'labels')

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader
# gts, 以编码形式进行标注，pot0-6:无效的，不透水面、建筑物、低矮制备、树木、汽车、背景，在测试时只考虑中间物种目标
train_dataset = VaihingenDataset(data_root='./data/vaihingen_512/train', mode='train',
                                img_dir='images_512', mask_dir='masks_512',
                                mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='./data/vaihingen_512/val', mode='val',
                                img_dir='images_512', mask_dir='masks_512', 
                                transform=val_aug)

test_dataset = VaihingenDataset(data_root='vaihingen_1024',mode='test',
                                img_dir='img/test', mask_dir='label/test', 
                                transform=val_aug)

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
                        drop_last=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=test_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=lr * 0.0001, T_0=15, T_mult=2)
# lr_scheduler = torch.optim.lr_scheduler.My_CosineAnnealingWarmRestarts(optimizer, eta_min=lr * 10e-8, 
#                                                                        T_0=15, T_mult=2, epoch_num = (max_epoch +1) * len(train_loader.dataset))


"""
optimizer:优化器，
T_0:学习率第一次回到初始学习率的位置, 15,45,90
"""

