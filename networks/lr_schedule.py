from config import config

def half_lr(init_lr, ep):
    lr = init_lr / 2**ep

    return lr

def step_lr(ep):
    if ep < 50:
        lr = 0.01
    elif ep < 120:
        lr = 0.001
    elif ep < 200:
        lr = 0.0001
    elif ep < 250:
        lr = 0.00001
    elif ep < 300:
        lr = 0.000005
    else:
        lr = 0.000001
    return lr


def warmup_lr(init_lr, warmup_epoch, epoch):
    if epoch < warmup_epoch:
        lr = init_lr / warmup_epoch * (epoch+1)
    elif epoch < warmup_epcoh + 80:
        lr = init_lr
    elif epoch < warmup_epoch + 120:
        lr = init_lr / 10
    elif epoch < warmup_epoch + 150:
        lr = init_lr / 100
    elif epoch < warmup_epoch + 200:
        lr = init_lr / 1000
    else:
        lr = 1e-5

    return lr
