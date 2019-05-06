# iDesigner - FGVCx 2019 - Hearst Magazine Media
竞赛官网: [https://www.kaggle.com/c/idesigner/overview](https://www.kaggle.com/c/idesigner/overview)
## Dataset
通过服装风格预测设计师

## ENVS
* Ubuntu == 16.04
* python == 2.7
* pytorch == 0.4.1 / 1.0.1
* visdom == 0.1.8.8

## File Structure
```
kaggle_idesigner/
▾ data/
    label_list.txt
    last_no_tta.csv
    result_3TTA.csv
    result_no_TTA.csv
    train.txt
    valid.txt
▾ dataset/
    __init__.py
    augmentation.py
    create_img_list.py
    dataset.py
    EDA.py
▾ metrics/
    __init__.py
    metric.py
▾ networks/ 
    __init__.py
    loss.py
    lr_schedule.py
    network.py
  __init__.py
  config.py
  inference.py
  README.md
  train.py
```

## RUN
* STEP0
```
git clone https://github.com/xungeer29/kaggle_idesigner
cd kaggle_idesigner
```
* STEP1
添加文件搜索路径，更改数据集根目录

将所有的`.py`文件的`sys.path.append`中添加的路径改为自己的项目路径

更改`config.py`中的`data_root`为数据集存放的根目录
* STEP2
划分训练集和本地验证集

```
python dataset/create_img_list.py
```

* STEP3
train

```
python train.py
```

* STEP4
inference
```
python inference.py
```

## TODO
* data distulation
* OHEM
* pretrainedmodels senet spp

## Experiments
* 全卷积的 ResNet18, Focal Loss, alpha=None, gamma=2, acc@train=95.89, acc@val=76.10, acc@notta=76.831, acc@3tta=78.396
* WeightSampler + ResNet152 acc@train=82.95, acc@val=72.4943, acc@notta=73.450, acc@3tta=74.389
* ResNet18, 2层全连接，WeightSampler, FocalLoss, acc@train=99.8438, acc@val=81.6006, acc@notta=85.472, acc@3tta=85.097
* LabelSmoothing: 损失为负，网络无法收敛, 怎么调节？
* ResNet18 + 2层 FC，WeightSampler, LabelSmoothing, acc@train
* ResNet18+2FC, WeightSampler, RandomErasing, acc@train=99.3359, acc@val=87.0394, acc@notta=90.169, acc@3tta=89.668
* 将第一层的 7\*7改为三个3\*3, acc@train=98.3594, acc@val=85.2742, acc@notta=88.227, 下降了，应该是自己训练的卷积层没有预训练的卷积层的效果好
* 在7\*7网络结构的最后一个FC后加上 l2_norm, acc@train=96.875, acc@notta=88.791, 下降了，是学习率的调节问题？

## Reference
* [Bag of Tricks for Image Classification with CNN](https://arxiv.org/abs/1812.01187)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)
