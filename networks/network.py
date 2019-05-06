# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pretrainedmodels

def whitening(im):
    batch_size, channel, h, w = im.shape
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = torch.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im

def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model

        # 3 3*3 convs replace 1 7*7
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        # conv
        # conv replace FC
        self.conv_final = nn.Conv2d(512, num_classes, 1, stride=1)
        self.ada_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x = whitening(x)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        '''
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # FC
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = l2_norm(x)

        '''
        # fully conv
        x = self.conv_final(x)
        x = self.ada_avg_pool(x)
        x = x.view(x.size(0), -1)
        # print x.size()
        '''


        return x

class ResNet34(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet34, self).__init__()
        self.backbone = model

        # FC
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = l2_norm(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        '''
        self.conv_final1 = nn.Conv2d(2048, 1024, 1, stride=1)
        self.relu = nn.ReLU()
        self.conv_final2 = nn.Conv2d(1024, num_classes, 1, stride=1)

        self.ada_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        '''

        # FC
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = l2_norm(x)

        return x


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet101, self).__init__()
        self.backbone = model

        '''
        self.conv_final1 = nn.Conv2d(2048, 1024, 1, stride=1)
        self.relu = nn.ReLU()
        self.conv_final2 = nn.Conv2d(1024, num_classes, 1, stride=1)

        self.ada_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        '''

        # FC
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = l2_norm(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet152, self).__init__()
        self.backbone = model

        self.conv_final1 = nn.Conv2d(2048, 1024, 1, stride=1)
        self.relu = nn.ReLU()
        self.conv_final2 = nn.Conv2d(1024, num_classes, 1, stride=1)

        self.ada_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # FC
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        '''
        # conv
        x = self.conv_final1(x)
        x = self.relu(x)
        x = self.conv_final2(x)
        x = self.relu(x)
        x = self.ada_avg_pool(x)
        x = x.view(x.size(0), -1)
        '''

        return x

class se_resnet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(se_resnet50, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone

if __name__ == '__main__':
    #'''
    backbone = models.resnet101(pretrained=True)
    models = ResNet101(backbone, 50)
    print models
    data = torch.randn(1, 3, 100, 300)
    x = models(data)
    #print(x)
    print(x.size())
    '''
    backbone = pretrainedmodels.__dict__['se_resnet50'](pretrained='imagenet')
    models = se_resnet50(backbone, 50)
    data = torch.randn(1, 3, 100, 300)
    x = models(data)
    #print(x)
    print(x.size())
    '''
