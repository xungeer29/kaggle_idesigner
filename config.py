# -*- coding:utf-8 -*- 

class DefaultConfigs(object):
    data_root = '/media/gfx/data1/DATA/Kaggle/idesigner' # 数据集的根目录
    model = 'ResNet50' # ResNet152 使用的模型
    freeze = True # 是否冻结卷基层

    seed = 1000 # 固定随机种子
    num_workers = 2 # DataLoader 中的多线程数量
    num_classes = 50 # 分类类别数
    num_epochs = 500
    batch_size = 64 # 128 48
    lr = 0.01 # 初始lr
    width = 100 # 输入图像的宽
    height = 300 # 输入图像的高
    iter_smooth = 10 # 打印&记录log的频率

    # OHEM
    OHEM = False
    OHEM_ratio = 0.1

    # resume checkpoint
    resume = False #
    checkpoint = 'ResNet50.pth' # 训练完成的模型名

    # smooth label
    smooth_label = False

config = DefaultConfigs()
