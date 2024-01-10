import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp   #多进程
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    """设置随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

"""lable匹配"""
ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
Invalid = np.array([0, 0, 0]) # label 6
num_classes = 6   #类别


# split huge RS image to small patches

def parse_args():
    """配置参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="/home/guest/Pycode/3/segmentation/data_vaihingen/val/image/")
    parser.add_argument("--mask-dir", default="/home/guest/Pycode/3/segmentation/data_vaihingen/val/label/")
    parser.add_argument("--output-img-dir", default="data/vaihingen_256/test/images_256")
    parser.add_argument("--output-mask-dir", default="data/vaihingen_256/test/masks_256")
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    return parser.parse_args()


def get_img_mask(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)   #转成数组格式
    img_pad = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    return img_pad, mask_pad


def pv2rgb(mask):
    """标签上色"""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def car_color_replace(mask):
    """给汽车换一种颜色"""
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask


def rgb_to_2D_label(_label):
    """one-hot"""
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Invalid, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    """增强"""
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        #只增强训练数据
        h_vlip = RandomHorizontalFlip(p=1)
        v_vlip = RandomVerticalFlip(p=1)

        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())

        image_list_train = [image, image_h_vlip, image_v_vlip]  #3倍数据
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]

        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask(image_list_train[i], mask_list_train[i], patch_size, mode)
            # image_tmp, mask_tmp = image_list_train[i], mask_list_train[i]
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask(image.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), width=h, height=w)(image=image.copy(), mask=mask.copy())
    img_crop, mask_crop = crop['image'], crop['mask']
    return img_crop, mask_crop


def car_aug(image, mask):
    """汽车增强"""
    assert image.shape[:2] == mask.shape
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
    # blur = albu.GaussianBlur(p=1.0)(image=image.copy())
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
    # blur_image = blur['image']
    image_list = [image, image_rotate]
    mask_list = [mask, mask_rotate]

    return image_list, mask_list


def vaihingen_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, mode, val_scale, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]+"_noBoundary"
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    #获取待分割图片的路径

    print("file name:", img_filename, mask_filename)
    if eroded:  #使用何种标签
        mask_path = mask_path[:-4] + '_noBoundary.tif'
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    if gt:
        mask_ = car_color_replace(mask)
        out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
        cv2.imwrite(out_origin_mask_path, mask_)

    # img and mask shape: WxHxC
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
                                          mode=mode, val_scale=val_scale)

    assert img_filename == mask_filename and len(image_list) == len(mask_list)
    #得到图像和标签文件

    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]  #保证图片和标签图大小一致
        if gt:   #是否将标签图转换成颜色图
            mask = pv2rgb(mask)

        for y in range(0, img.shape[0], stride):    #开始切割
            for x in range(0, img.shape[1], stride):
                if (y+split_size)>img.shape[0]:
                    y = img.shape[0] - split_size

                if (x+split_size)>img.shape[1]:
                    x = img.shape[1] - split_size
                img_tile = img[y:y + split_size, x:x + split_size]
                mask_tile = mask[y:y + split_size, x:x + split_size]

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:
                    image_crop, mask_crop = randomsizedcrop(img_tile, mask_tile)
                    #随机裁剪缩放
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)    #各类别像素数目
                    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])   #各类别像素数目在总像素中所占的比例
                    if cf[4] > 0.1 and mode == 'train':    #当每张训练图像中，汽车像素数目不足10%，进行二次增强
                        car_imgs, car_masks = car_aug(image_crop, mask_crop)
                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(imgs_output_dir,
                                                        "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
                            cv2.imwrite(out_img_path, car_imgs[i])

                            out_mask_path = os.path.join(masks_output_dir,
                                                         "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                            cv2.imwrite(out_mask_path, car_masks[i])
                    else:
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    gt = args.gt
    eroded = args.eroded
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride

    #glob:可以查找符合要求的文件
    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))
    if eroded:
        mask_paths = [(p[:-15] + '.tif') for p in mask_paths_raw]
    else:
        mask_paths = mask_paths_raw
    img_paths.sort()
    # mask_paths.sort()
    for i in range(len(mask_paths)):
        mask_paths[i] = img_paths[i].replace('image', 'label')
        mask_paths[i] = mask_paths[i].replace('.tif', '_noBoundary.tif')


    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir+'/origin')

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


