
from skimage import io
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import numpy as np
import os


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

    img = io.imread('/home/guest/Pycode/3/GeoSeg-main/data/vaihingen/train/masks_512/top_mosaic_09cm_area7_noBoundary_0_1_0.png')
    print(img)
    img = label2rgb(img)
    plt.figure() 
    plt.imshow(img) # 原始图片
    plt.show()

def count_dir():
    dir = '/home/guest/Pycode/3/GeoSeg-main/data/vaihingen/train/images_512'
    dir_list = os.listdir(dir)
    print(len(dir_list))

if __name__ == '__main__':
    count_dir()



