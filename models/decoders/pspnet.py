
import torch
import torch.nn as nn
import functools

from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d


__all__ = ['PyramidalPooling', 'PSPNet']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class PyramidalPooling(nn.Module):
    def __init__(self, fc_dim=2048, pool_scales=(1, 2, 3, 6)):
        super(PyramidalPooling, self).__init__()
        #self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)


    def forward(self, conv_in):
        '''
        '''

        input_size = conv_in.size()
        ppm_out = [conv_in]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv_in),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)

        return ppm_out


class PSPNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 inter_channels= 512):
        super(PSPNet, self).__init__()

        self.pyramid= PyramidalPooling(fc_dim=2048, pool_scales=(1, 2, 3, 6))
        self.conv_last = nn.Sequential(
                            nn.Conv2d(fc_dim * 2, inter_channels, 3, padding=1, bias=False),
                            BatchNorm2d(inter_channels),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(0.1, False),
                            nn.Conv2d(inter_channels, num_class, 1))

    def forward(self, conv_out, outSize=None):
        '''
        '''
        conv5 = conv_out[-1]
        ppm_out= self.pyramid(conv5)
        x = self.conv_last(ppm_out)

        if outSize is not None:  # is True during inference
            x = nn.functional.interpolate(
                x, size=outSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)
        return x

