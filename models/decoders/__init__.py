import torch.nn as nn
# from .upernet import UPerNet
from .pspnet import PSPNet
from .ocnet import OCNet
from .danet import DANet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


def buildDecoder(name, num_classes, weights=''):
    if name == 'pspnet':
        decoder= PSPNet(num_class= num_classes)
    elif name == 'ocnet':
        decoder = OCNet(num_classes= num_classes)
    elif name == 'danet':
        decoder = DANet(num_classes=num_classes)
    # elif name == 'upernet':
    #     decoder = UPerNet(num_class= num_classes)
    else:
        raise NotImplementedError('The provided decoder is not Implemented')

    if len(weights) > 0:
        print('Loading weights for net_decoder')
        decoder.load_state_dict(
            torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
    else:
        decoder.apply(weights_init)
    return decoder
