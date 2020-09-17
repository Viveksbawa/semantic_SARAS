import torch.nn as nn
from .encoder import Resnest, ResnetDilated
from .ResNest import resnest50, resnest101, resnest200
from .resnet2 import resnet50, resnet101


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


def buildEncoder(name= 'resnest200', weights= ''):
    if name == 'resnest200':
        model= resnest200(pretrained=True)
        new_model = Resnest(orig_net=model)
    elif name == 'resnest101':
        model= resnest101(pretrained=True)
        new_model= Resnest(orig_net=model)
    elif name == 'resnet101':
        model= resnet101(pretrained=True)
        new_model= ResnetDilated(orig_resnet=model)
    elif name == 'resnet50':
        model= resnet50(pretrained=True)
        new_model= ResnetDilated(orig_resnet=model)
    else:
        raise NotImplementedError('The encoderis not implemented')

    if len(weights) > 0:
        print('Loading weights for net_encoder')
        new_model.load_state_dict(
            torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

    return new_model
