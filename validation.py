import numpy as np
import torch
from tqdm import tqdm
from datasets import ade20k
from torch.nn.parallel.scatter_gather import gather
from utils import batch_intersection_union, batch_pix_accuracy, AverageMeter


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
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
    # print('Validation IOU:{:.5f} === ACC:{:.5f}'.format(miou, acc * 100))
    return mIoU, IoU, pixAcc



