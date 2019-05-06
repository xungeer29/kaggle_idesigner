# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/kaggle_idesigner')
import os, argparse, time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, models
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import visdom
import pretrainedmodels

from dataset.dataset import *
# from dataset.augmentation import *
from networks.network import *
from networks.lr_schedule import *
from metrics.metric import *
from config import config
from networks.loss import FocalLoss, LabelSmoothing

def create_weights(dataset, label_list):
    label2name = {}
    with open(label_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        name, label = line.strip().split(',')
        label2name[int(label)] = name
    name2num = {}
    path = os.path.join(config.data_root, 'designer_image_train_v2_cropped', 'designer_image_train_v2_cropped')
    for name in os.listdir(path):
        name2num[name] = len(os.listdir(os.path.join(path, name)))

    weights = []
    for im, label in dataset:
        name = label2name[label]
        num = name2num[name]
        weights.append(1./num)

    return weights
    

def train():
    # model
    if config.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        model = ResNet50(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        model = ResNet101(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        model = ResNet152(backbone, num_classes=config.num_classes)
    elif config.model == 'se_resnet50':
        backbone = pretrainedmodels.__dict__['se_resnet50'](pretrained='imagenet')
        model = se_resnet50(backbone, num_classes=config.num_classes)
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    print model
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # freeze layers
    if config.freeze:
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        #for p in model.backbone.layer4.parameters(): p.requires_grad = False


    # loss
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = FocalLoss(config.num_classes, alpha=None, gamma=2, size_average=True)
    # criterion = LabelSmoothing(config.num_classes, 0, 0.1)

    # train data
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(10),
                                    transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = iDesignerDataset('./data/train.txt', transform=transform)
    weights = create_weights(dst_train, './data/label_list.txt')
    sampler = WeightedRandomSampler(weights, num_samples=len(dst_train), replacement=True)
    dataloader_train = DataLoader(dst_train, shuffle=False, batch_size=config.batch_size, 
                                  num_workers=config.num_workers, sampler=sampler)

    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = iDesignerDataset('./data/valid.txt', transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config.batch_size/2, num_workers=config.num_workers)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-'*30+'\n')
    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
               config.model, config.num_classes, config.num_epochs,
               config.width, config.height, config.iter_smooth))

    # load checkpoint
    if config.resume:
        print 'resume checkpoint...'
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))

    # visdom
    vis = visdom.Visdom(env='kaggle_idesigner')

    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
    #                              lr=0.00001, betas=(0.9, 0.999), weight_decay=0.0002)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                       lr=lr, momentum=1e-1, weight_decay=1e-4)

    # adjust lr
    # lr = half_lr(config.lr, epoch)
    # lr = step_lr(epoch)
    # lr_scheduler = torch.optim.lr_scheduler.
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 100, 150, 200], gamma=0.1)

    cudnn.benchmark = True

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    max_val_top1_acc = 0
    iters = 0
    for epoch in range(config.num_epochs):
        ep_start = time.time()
        lr = step_lr(epoch)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=0.01, betas=(0.9, 0.999), weight_decay=0.0002)
        model.train()
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda().long()

            output = model(input)

            if config.smooth_label:
                smoothed_target = label_smoothing(output, target).cuda()
                loss = F.kl_div(output, smoothed_target).cuda()
            
            # OHEM: online hard example mining
            if not config.OHEM and not config.smooth_label:
                loss = criterion(output, target)
            elif config.OHEM:
                if epoch < 50:
                    loss = criterion(output, target)
                else:
                    loss = F.cross_entropy(output, target, reduce=False).cuda()
                    OHEM, _ = loss.topk(int(config.num_classes*config.OHEM_ratio), dim=0, 
                                        largest=True, sorted=True)
                    loss = OHEM.mean()

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()

            acc = accuracy(output.data, target.data, topk=(1,))
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += acc[0]
            sum += 1


            if (i+1) % config.iter_smooth == 0:
                iters += 1
                vis.line(X=torch.FloatTensor([iters]), Y=torch.FloatTensor([train_loss_sum/sum]),
                         win='train_loss', update='append')
                vis.line(X=torch.FloatTensor([iters]), Y=torch.FloatTensor([train_top1_sum/sum]),
                         win='train_acc_top1', update='append')

                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, top1: %.4f'
                       %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                        train_loss_sum/sum, train_top1_sum/sum))
                log.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, top1: %.4f\n'
                           %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                            train_loss_sum/sum, train_top1_sum/sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0
 
        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < config.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1  = eval(model, dataloader_valid, criterion)
            val_time = (time.time() - val_time_start) / 60.

            vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([val_loss]),
                     win='val_loss', update='append')
            vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([val_top1]),
                     win='val_acc_top1', update='append')

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, best_top1: %.4f'
                   %(epoch+1, config.num_epochs, val_loss, val_top1, max_val_top1_acc))
            print('epoch time: {} min'.format(epoch_time))
            if val_top1[0].data > max_val_top1_acc:
                max_val_top1_acc = val_top1[0].data
                print('Taking top1 snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}.pth'.format('checkpoints', config.model))

            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, best_top1: %.4f\n'
                       %(epoch+1, config.num_epochs, val_loss, val_top1, max_val_top1_acc))
        torch.save(model, '{}/{}_last.pth'.format('checkpoints', config.model))

    log.write('-'*30+'\n')
    log.close()

# validation
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        acc_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += acc_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1

if __name__ == '__main__':
    '''
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = iDesignerDataset('./data/train.txt', transform=transform)
    weights = create_weights(dst_train, './data/label_list.txt')
    print weights, len(weights)
    '''
    train()
