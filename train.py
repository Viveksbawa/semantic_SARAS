import os
import time
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather
# Our libs
from config import cfg
from datasets import ade20k
from models import encoders, decoders
from models import SegmentationModel2

from utils import AverageMeter, checkpoint, batch_pix_accuracy
from functions import LR_Scheduler, validation
from lib.parallel import DataParallelModel, DataParallelCriterion
from lib.loss import SegmentationLosses

#----------------------------------------------------------------------------
#Training setup and configuration loading
parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
parser.add_argument("--cfg", default="config/ade20k-resnet.yaml", metavar="FILE",
    help="path to config file", type=str)

args = parser.parse_args()
cfg.merge_from_file(args.cfg)
#cfg.freeze()

# Output directory and current config writing
if not os.path.isdir(cfg.DIR):
    os.makedirs(cfg.DIR)
with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
    f.write("{}".format(cfg))

random.seed(cfg.TRAIN.seed)
torch.manual_seed(cfg.TRAIN.seed)

#-----------------------------------------------------------------------
# Tarining and Validation Dataset Loader
dataset_train = ade20k.ADE20KSegmentation(root= cfg.DATASET.root_dataset, 
                split='train', scale= cfg.DATASET.multiscale)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.TRAIN.batch_size,
    shuffle=True, num_workers=cfg.TRAIN.workers, drop_last=True, pin_memory=True)

if cfg.VAL.validate:
    dataset_val = ade20k.ADE20KSegmentation(
                    root= cfg.DATASET.root_dataset, split='val')
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.TRAIN.batch_size, 
                shuffle=False, num_workers=cfg.TRAIN.workers, drop_last=False)

#Model definition
print('Building model with Encoder:{} and Decoder:{}'.format(
        cfg.MODEL.arch_encoder, cfg.MODEL.arch_decoder))
net_encoder= encoders.buildEncoder(name=cfg.MODEL.arch_encoder,
                                    weights=cfg.MODEL.weights_encoder)

net_decoder= decoders.buildDecoder(name= cfg.MODEL.arch_decoder,
                                    num_classes=cfg.DATASET.num_class,
                                    weights=cfg.MODEL.weights_decoder)


model = SegmentationModel2(net_enc=net_encoder, net_dec= net_decoder)
# criterion = OhemCrossEntropy2d(ignore_label=-1, use_weight=False)
criterion = SegmentationLosses()

# optimizer and scheduler 
params_list = [{'params': model.encoder.parameters(), 'lr': cfg.TRAIN.lr_encoder},
                {'params': model.decoder.parameters(), 'lr': cfg.TRAIN.lr_encoder * 10}]

optimizer = torch.optim.SGD(params_list, lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.momentum, weight_decay=cfg.TRAIN.weight_decay)

scheduler = LR_Scheduler(mode='poly', base_lr= cfg.TRAIN.lr_encoder, 
            num_epochs= cfg.TRAIN.num_epoch,iters_per_epoch= len(loader_train))

# print(list(model.modules()))
# transfer model copies to the gpus
model = DataParallelModel(model).cuda()
criterion = DataParallelCriterion(criterion).cuda()


#--------------------------------------------------------------------------------
# Model training
history= {'tr_loss':[], 'tr_acc':[],
            'val_miou':[], 'val_ciou':[], 'val_acc':[]}

best_pred = [0.0, 0.0]
for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    model.train()
    # model.train(not cfg.TRAIN.fix_bn)
    tic = time.time()
    for i, (imgs, labels) in enumerate(loader_train):
        lr= scheduler(optimizer, i, epoch)
        optimizer.zero_grad()
        outputs= model(imgs, outSize=(480, 480))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item())
        output_comb = gather(outputs, 0, dim=0)
        pred= output_comb[0]
        tr_acc, tr_pix= batch_pix_accuracy(pred.data, labels.cuda())
        train_acc.update(tr_acc/tr_pix)
        # print(train_loss.average(), train_acc.average())
        # print(tr_acc/tr_pix, loss.item())
        
        #print training information
        print('\rTraining:Epoch[{}]-Iter[{}/{}] = Loss:{:.3f}' 
            '= Acc:{:.3f} = LR:{:.5f} = time:{:.1f}'.format(
            epoch, i, len(loader_train), train_loss.average(), train_acc.average(), lr,
            time.time() - tic), end= '')

    history['tr_loss'].append(train_loss.average())
    history['tr_acc'].append(train_acc.average())
    
    # validation of model performance in the validation set
    if cfg.VAL.validate:
        if (epoch+1) % cfg.VAL.val_step ==0 or epoch+1 == cfg.TRAIN.num_epoch:
            print('Evaluating model!')
            val_miou, val_ciou, val_acc = validation(model, loader_val, cfg)
            history['val_miou'].append(val_miou)
            history['val_ciou'].append(val_ciou)
            history['val_acc'].append(val_acc)

            if val_miou > best_pred[0]:
                best_pred[0] = val_miou
                best_model= True
            else:
                best_model=False
            if val_acc > best_pred[1]:
                best_pred[1] = val_acc
            print('Best validation IOU and Acc:', best_pred)

            checkpoint(state={'epoch': epoch,
                        'encoder': net_encoder.state_dict(),
                        'decoder': net_decoder.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        cfg=cfg, best= best_model, history= history)


    







