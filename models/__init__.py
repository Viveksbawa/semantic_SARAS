import torch
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, net_enc, net_dec, crit, use_aux=False):
        super(SegmentationModel, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.use_aux = use_aux

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, imgs, labels= None, outSize=None):
        if outSize is None:
            pred = self.decoder(self.encoder(imgs, return_feature_maps=True),
                outSize=outSize)
            loss= self.crit(pred, labels)
            acc= self.pixel_acc(pred, labels)
            return loss, acc
        else:
            pred = self.decoder(self.encoder(imgs, return_feature_maps=True),
                    outSize=outSize)
        return pred


class SegmentationModel2(nn.Module):
    def __init__(self, net_enc, net_dec):
        super(SegmentationModel2, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec

    def forward(self, imgs, outSize=None):
        pred = self.decoder(self.encoder(imgs, return_feature_maps=True),
                outSize=outSize)
        x=list()
        x.append(pred)
        return tuple(x)


