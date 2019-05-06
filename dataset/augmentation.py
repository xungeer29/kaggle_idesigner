from __future__ import absolute_import
import sys
sys.path.append('/home/gfx/Projects/kaggle_idesigner')

from PIL import Image, ImageFilter
from torchvision.transforms import *
import random
import math
import numpy as np
import torch

def RandomErasing(im, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[128, 128, 128]):
    '''
    performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    img: PIL img
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    if random.uniform(0, 1) > probability:
        return im

    else:
        img = np.array(im)
        area = img.shape[0] * img.shape[1]
       
        while True:
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if img.shape[0] > w and img.shape[1] > h:
                break

        if w < img.shape[0] and h < img.shape[1]:
            x1 = random.randint(0, img.shape[0] - w)
            y1 = random.randint(0, img.shape[1] - h)
            # if img.size()[0] == 3:
            if im.mode == 'RGB':
                img[x1:x1+h, y1:y1+w, 0] = mean[0]
                img[x1:x1+h, y1:y1+w, 1] = mean[1]
                img[x1:x1+h, y1:y1+w, 2] = mean[2]
            elif im.mode == 'L':
                img[x1:x1+h, y1:y1+w] = mean[0]
        img = Image.fromarray(np.uint8(img))

        return img


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)

if __name__ == '__main__':
    # test MyGaussianBlur
    path = r'/media/gfx/data1/DATA/Kaggle/idesigner/designer_image_train_v2_cropped/designer_image_train_v2_cropped/alexander mcqueen/FW08DLR_McQueen_0014.png'
    im = Image.open(path)
    # im = im.filter(MyGaussianBlur(radius=5))
    # im.save('./figs/GaussianBlur.jpg')
    
    im = RandomErasing(im, probability=0.5)
    im.save('randomerasing.jpg')
