import numpy as np
import torch
# import math
import torch.nn as nn
from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

__all__ = ['PAM_Module', 'CAM_Module']




class PAM_Module(nn.Module):
    """ Position attention module
        in_dim: number of channels in input feature map
    """
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        sasc_output = self.conv8(feat_sum)

        # return tuple(sasc_output, sa_output, sc_output)
        return sasc_output



class DANet(nn.Module):
    """Fully Convolutional Networks for Semantic Segmentation

    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, num_classes, fc_dim=2048, use_aux=False):
        super(DANet, self).__init__()
        self.use_aux = use_aux
        self.da_head = DANetHead(fc_dim, num_classes)
        if self.use_aux:
            self.aux_layer = nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                    BatchNorm2d(512),
                    nn.Dropout2d(0.05),
                    nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                    )            

    def forward(self, conv_out, outSize=None):

        conv5= conv_out[-1]
        imsize = conv5.size()[2:]

        x = self.da_head(conv5)
        # x = list(x)
        # x[0] = nn.functional.interpolate(x[0], size=imsize, mode='bilinear', align_corners=False)
        # x[1] = nn.functional.interpolate(x[1], size=imsize, mode='bilinear', align_corners=False)
        # x[2] = nn.functional.interpolate(x[2], size=imsize, mode='bilinear', align_corners=False)
        # return tuple(x[0], x[1], x[2])

        if outSize is not None:  # is True during inference
            x = nn.functional.interpolate(
                x, size=outSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)
        if not self.use_aux:
            return x
        else:
            aux_out= self.aux_layer(conv_out[-2])
            aux_out = nn.functional.log_softmax(aux_out, dim=1)
            return (x, aux_out)



