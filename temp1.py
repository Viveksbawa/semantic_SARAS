import numpy as np
import torch
from models import decoders, encoders
from functions import group_weight

dec_model = decoders.buildDecoder(name='danet', num_classes= 150)
# enc_model = encoders.buildEncoder(name='resnest200')
# data = torch.Tensor(np.random.rand(2, 3, 400, 400))
# out1= enc_model(data, return_feature_maps=True)
# print(out1[0].shape, out1[1].shape, out1[2].shape, out1[3].shape)

# out2= dec_model(out1)
# print(out2.shape)
# from models.decoders import danet
from torch import nn
a= nn.Parameter(torch.zeros(1))
# if isinstance(a, nn.Parameter):
#     print('yes')
# model= danet.CAM_Module(2048)
# print(model.parameters)
# for m in model.modules():
#     print(m)

