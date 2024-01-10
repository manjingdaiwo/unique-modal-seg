from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from models.lanet import LANet as Net
# from catalyst.contrib.nn import Lookahead
# from catalyst import utils
from datetime import datetime
from IPython import embed

# training hparam
max_epoch = 2
ignore_index = len(CLASSES)

train_batch_size = 2
val_batch_size = 2
test_batch_size = 2

lr = 6e-4
weight_decay = 0.01

backbone_lr = 6e-5
backbone_weight_decay = 0.01

accumulate_n = 4
num_classes = len(CLASSES)
classes = CLASSES

save_per_epoch = 5

gpu = True
gpus = [0]

pretrained_ckpt_path = None

#  define the network
#net = UNetFormer(num_classes=num_classes)
net = Net(num_classes=num_classes)
NET_NAME = 'lanet'
DATA_NAME = 'Vaihingen'
working_path = '/home/guest/Pycode/3/GeoSeg-main'


DATE_FORMAT = '%d_%B_%Hh_%Mm'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
chkpt_dir_best = os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, TIME_NOW, 'best')
chkpt_dir_regular = os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, TIME_NOW, 'regular')
log_dir = os.path.join(working_path, 'logs', NET_NAME, DATA_NAME,  TIME_NOW)

chkpt_dir_best_m = os.path.join(working_path, 'checkpoints_m', NET_NAME, DATA_NAME, 'best')
chkpt_dir_regular_m = os.path.join(working_path, 'checkpoints_m', NET_NAME, DATA_NAME, 'regular')
log_dir_m = os.path.join(working_path, 'logs_m', NET_NAME, DATA_NAME)

pred_dir_preds = os.path.join(working_path, 'results',  NET_NAME, DATA_NAME, 'preds')
pred_dir_labels = os.path.join(working_path, 'results',  NET_NAME, DATA_NAME, 'labels')

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader
train_dataset = VaihingenDataset(data_root='./data/vaihingen_256/test', mode='train',
                                img_dir='images_256', mask_dir='masks_256',
                                mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='./data/vaihingen_256/test', mode='val',
                                img_dir='images_256', mask_dir='masks_256', 
                                transform=val_aug)

test_dataset = VaihingenDataset(data_root='./data/vaihingen_256/test',mode='test',
                                img_dir='images_256', mask_dir='masks_256', 
                                transform=val_aug)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=test_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

# optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=lr * 0.0001, T_0=15, T_mult=2)

