# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/kaggle_idesigner')

import os
import random
from config import config
from tqdm import tqdm

random.seed(config.seed)

if not os.path.exists('./data'):
    os.makedirs('./data')

train_txt = open('./data/train.txt', 'w')
val_txt = open('./data/valid.txt', 'w')
label_txt = open('./data/label_list.txt', 'w')

label_list = []
train_path = os.path.join(config.data_root, 'designer_image_train_v2_cropped/designer_image_train_v2_cropped')
for dir in tqdm(os.listdir(train_path)):
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{},{}\n'.format(dir, str(len(label_list)-1)))
        data_path = os.path.join(train_path, dir)
        train_list = random.sample(os.listdir(data_path), 
                                   int(len(os.listdir(data_path))*0.95))
        for im in train_list:
            train_txt.write('{}/{}/{},{}\n'.format(train_path, dir, im, str(len(label_list)-1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{}/{},{}\n'.format(train_path, dir, im, str(len(label_list)-1)))
