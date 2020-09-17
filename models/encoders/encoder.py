#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:58:52 2020

@author: vivek
"""

import torch
import torch.nn as nn
#import torchsummary

from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d


#Encoder definitions -----------------------------------------------------------------
# Encoder for the ResNest models
# To change the stride and dilation please check Resnet/resnet.py file: class ResNet
class Resnest(nn.Module):
    def __init__(self, orig_net):
        super(Resnest, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_net.conv1
        self.bn1 = orig_net.bn1
        self.relu1 = orig_net.relu
        self.maxpool = orig_net.maxpool

        self.layer1 = orig_net.layer1
        self.layer2 = orig_net.layer2
        self.layer3 = orig_net.layer3
        self.layer4 = orig_net.layer4
    

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


#--------------------------------------------------------------------------------------
#Resnet encoder with dilation
class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);
        if return_feature_maps:
            return conv_out
        return [x]



# if __name__ == '__main__':
#     import numpy as np

#     model= resnet50(pretrained=True)
#     new_model= Resnet(model)
#    model= buildEncoder(name= 'resnet50Dilated')
#    #torchsummary.summary(model=model, input_size=(3,200,200))
#    inp= torch.Tensor(np.random.rand(2,3,200,200))
#    out= new_model(inp)
#    print(out[0].shape)

