import math
import numpy as np
import torch
from tqdm import tqdm
# from datasets import ade20k
from torch.nn.parallel.scatter_gather import gather
from utils import batch_intersection_union, batch_pix_accuracy, AverageMeter

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step

        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented

        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10





# Validation function
def validation(model, val_loader, cfg):
    nclass= cfg.DATASET.num_class
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    # model evaluation on validation set 
    model.eval()
    tbar = tqdm(val_loader, desc='\r')
    for i, (image, target) in enumerate(tbar):
        with torch.no_grad():
            outputs = model(image, outSize=(480, 480))
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = batch_pix_accuracy(pred.data, target)
            inter, union = batch_intersection_union(pred.data, target, nclass)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union

            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'Validation mIoU: %.3f, pixAcc: %.3f' % (mIoU, pixAcc))
    return mIoU, IoU, pixAcc


