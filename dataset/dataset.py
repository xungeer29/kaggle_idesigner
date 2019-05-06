# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/kaggle_idesigner')
import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from augmentation import MyGaussianBlur, RandomErasing

def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(',')
            ims.append(im)
            labels.append(int(label))
    return ims, labels

class iDesignerDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.ims, self.labels = read_txt(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        im_path = self.ims[index]
        label = self.labels[index]
        im_path = os.path.join(config.data_root, im_path)
        im = Image.open(im_path)
        #im = im.resize((self.width, self.height))
        if self.transform is not None:
            im = RandomErasing(im, probability=0.5, sl=0.1, sh=0.4, r1=0.3, mean=[128, 128, 128])
            if random.random() < 0.5:
                im = im.filter(MyGaussianBlur(radius=5))
            im = self.transform(im)

        return im, label

    def __len__(self):
        return len(self.ims)

def padding_image(im):
    w, h = im.size
    pad_im = Image.new('RGB', (w, h), (128, 128, 128))
    pad_im.paste(im, (0, 0))

    return pad_im

class iDesignerTestDataset(Dataset):
    def __init__(self, csv_path, transform=None, augment=None):
        df = pd.read_csv(csv_path)
        self.ims = df['Id']
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        name = self.ims[index]
        im_path = os.path.join(config.data_root, 'test', name)
        # print(im_path)
        im = Image.open(im_path)
        im = padding_image(im)
        #im = im.resize((self.width, self.height))
        if self.augment == 2:
            im = im
        # top-left crop
        if self.augment == 1:
            w, h = im.size
            im = im.crop((0, 0, w-10, h-10))
        # center crop
        if self.augment == 0:
            w, h = im.size
            im = im.crop((10, 10, w-10, h-10))

        if self.transform is not None:
            im = self.transform(im)

        return im, name

    def __len__(self):
        return len(self.ims)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = iDesignerDataset('./data/train.txt', transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    #for im, loc, cls in dataloader_train:
    for data in dataloader_train:
        print data
        #print loc, cls
    
