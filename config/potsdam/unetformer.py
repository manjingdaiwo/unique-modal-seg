from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from datetime import datetime
from models.model_utnet.utnet import UTNet, UTNet_Encoderonly

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
test_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None

save_per_epoch = 5

gpu = True
gpus = [0]

pretrained_ckpt_path = None

#  define the network
# net = UNetFormer(num_classes=num_classes)
net = UTNet(3, 32, num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True).cuda()


NET_NAME = 'utnet'
DATA_NAME = 'pot'
working_path = '/home/a1/ss/semantic_segmentation/my_segmodel'


DATE_FORMAT = '%d_%B_%Hh_%Mm'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
chkpt_dir_best = os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, TIME_NOW, 'best')
chkpt_dir_regular = os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, TIME_NOW, 'regular')
log_dir = os.path.join(working_path, 'logs', NET_NAME, DATA_NAME,  TIME_NOW)

pred_dir_preds = os.path.join(working_path, 'results',  NET_NAME, DATA_NAME, 'preds')
pred_dir_labels = os.path.join(working_path, 'results',  NET_NAME, DATA_NAME, 'labels')

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

train_dataset = PotsdamDataset(data_root='Potsdam_512', mode='train',
                                img_dir='irrg/train', mask_dir='gts/train',
                                mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(data_root='Potsdam_512', mode='val',
                                img_dir='irrg/valid', mask_dir='gts/valid', 
                                transform=val_aug)

test_dataset = PotsdamDataset(data_root='Potsdam_512',mode='test',
                                img_dir='irrg/test', mask_dir='gts/test', 
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

# define the optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=lr * 0.0001, T_0=15, T_mult=2)

