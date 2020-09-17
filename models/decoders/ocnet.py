import torch
# import os
# import sys
# import pdb
import numpy as np
from torch import nn
# import functools

from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

__all__ =['SelfAttentionBlock', 'BaseOC_Module', 'OCNet']


class SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels

        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = nn.functional.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = nn.functional.interpolate(
                    context, size=(h, w), mode='bilinear', align_corners=False)

        return context



class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, in_channels= 512, out_channels= 512, 
            key_channels= 256, value_channels= 256, dropout= 0.05, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(
                    in_channels, out_channels, key_channels, value_channels, size) for size in sizes])        
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels, kernel_size=1, padding=0),
            BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels, 
                                    size)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output



class OCNet(nn.Module):
    def __init__(self, num_classes, fc_dim=2048, use_aux=False, in_channels= 512, out_channels= 512, 
            key_channels= 256, value_channels= 256, dropout= 0.05, sizes=([1])):
        super(OCNet, self).__init__()
        
        self.use_aux = use_aux
        self.baseOC= BaseOC_Module(in_channels= 512, out_channels= 512, 
            key_channels= 256, value_channels= 256, dropout= 0.05, sizes=([1]))
        self.conv_in = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512))
        self.last_conv= nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        if self.use_aux:
            self.aux_layer = nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                    BatchNorm2d(512),
                    nn.Dropout2d(0.05),
                    nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                    )
    

    def forward(self, conv_out, outSize=None):
        conv5= conv_out[-1]

        x= self.conv_in(conv5)
        x= self.baseOC(x)
        x= self.last_conv(x)
        x = nn.functional.interpolate(x, size=outSize, mode='bilinear', align_corners=False)
        return x


