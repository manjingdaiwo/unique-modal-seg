import os
import cv2
import numpy as np


data_dir = 'aaa.png'


def handl_Canny(pic_path=data_dir):
    img=cv2.imread(pic_path)
    img=cv2.resize(src=img,dsize=(512,512))
    img_canny=cv2.Canny(img,threshold1=50,threshold2=200)
    cv2.imshow('image',img_canny)
    cv2.imshow('src',img)
    cv2.waitKey(0)

if __name__ == '__main__':
    print('PyCharm')
    handl_Canny()
