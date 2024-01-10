import multiprocessing.pool as mpp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from collections import OrderedDict
from tools.cfg import py2cfg
from torchvision.models import resnet50

from torch import nn
from torch.backends import cudnn
cudnn.benchmark = True
from tqdm import tqdm


# model = resnet50().to(device)


# dummy_input = torch.rand(1, 3, 256, 256).to(device)

# # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
# print('warm up ...\n')
# with torch.no_grad():
#     for _ in range(100):
#         _ = model(dummy_input)


torch.cuda.synchronize()

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 255, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb

def get_newest_file(path_file):
    path_file = str(path_file)
    lists = os.listdir(path_file)
    lists.sort(key=lambda fn: os.path.getmtime(path_file +'/'+fn))
    file_newest = os.path.join(path_file,lists[-2])
    return file_newest

def img_writer(inp):
    (mask,  mask_id) = inp

    mask_name_tif = mask_id + '.png'
    mask_tif = label2rgb(mask)
    cv2.imwrite(mask_name_tif, mask_tif)

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg('-m', "--multi_gpu", action = 'store_true', help='whether use multi gpu train the mdoel')

    return parser.parse_args()

args = get_args()
config = py2cfg(args.config_path)

if not os.path.exists(config.pred_dir_preds): 
    os.makedirs(config.pred_dir_preds)
if not os.path.exists(config.pred_dir_labels): 
    os.makedirs(config.pred_dir_labels)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def main():
    model = config.net
    model.cuda(config.gpus[0])   #测试时只使用一块gpu
    pth = get_newest_file("checkpoints/lanet/vai/07-05-23:18/best")
    if args.multi_gpu:
        state_dict = torch.load(pth)
        state_dict = state_dict['model_state_dict']
        checkpoint = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`            
            checkpoint[name] = v
            # load params
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['model_state_dict'])

    evaluator = Evaluator(num_class=config.num_classes)   #评价
    evaluator.reset()
    model.eval()
    print(pth+' Model loaded.')
    test_loader = config.test_loader


    repetitions = len(test_loader.dataset)
    timings = np.zeros((repetitions, 1))

    with torch.no_grad():
        print('testing ...\n')
        results, labels = [], []
        k = 0
        for input in tqdm(test_loader, total =len(test_loader), colour='GREEN'):
            starter.record()
            raw_predictions = model(input['img'].cuda(config.gpus[0]))
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[k] = curr_time
            k += 1

            raw_predictions = raw_predictions
            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                label = masks_true[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=label)
                mask_name = image_ids[i]
                results.append((mask, str(config.pred_dir_preds + '/' + mask_name)))
                labels.append((label, str(config.pred_dir_labels + '/' + mask_name)))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    print('-----------------------------per class----------------------------------')
    for class_name, class_iou, class_f1 in zip(config.classes[:-1], iou_per_class[:-1], f1_per_class[:-1]):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, np.round(class_f1, 4), class_name, np.round(class_iou, 4)))
        
    print("------------------------------overall-----------------------------------")
    print('F1:{}, mIOU:{}, OA:{}'.format(np.round(np.nanmean(f1_per_class[:-1]), 4), np.round(np.nanmean(iou_per_class[:-1]), 4), np.round(OA, 4)))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, labels)
    t1 = time.time()
    img_write_time = np.round((t1 - t0), 2)
    print('images writing spends: {} s'.format(img_write_time))
    avg = timings.sum()/(repetitions*1000)
    print('\navg time={}s\n'.format(avg))

if __name__ == "__main__":
    main()
