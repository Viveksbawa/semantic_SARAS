import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import torch.utils.data as data

class ADE20KSegmentation(data.Dataset):
    NUM_CLASS = 150
    def __init__(self, root, split='train', mode=None, img_transform=None, 
        target_transform=None, base_size=520, crop_size=480, scale=False,
        logger=None, **kwargs):

        super(ADE20KSegmentation, self).__init__()
        # assert exists and prepare dataset automatically
        self.root = os.path.join(root, 'ADEChallengeData2016')
        self.split= split
        self.mode = mode if mode is not None else split        
        assert os.path.exists(root), "Please setup the dataset using" + \
            "encoding/scripts/prepare_ade20k.py"
        
        self.img_transform = img_transform
        self.target_transform = target_transform

        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        self.input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
            
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}, multiscale {}'. \
                format(base_size, crop_size, scale))

            
        self.images, self.masks = _get_ade20k_pairs(self.root, split)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of:" + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # if self.mode == 'test':
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, os.path.basename(self.images[index])

        mask = Image.open(self.masks[index])

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
            
        # general resize, normalize and toTensor
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        img= self.input_transform(img)
        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        if self.scale:
            short_size = random.randint(int(self.base_size*0.95), int(self.base_size*1.6))
        else:
            short_size = self.base_size
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)#pad 255 for cityscapes

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        return img, self._mask_transform(mask)

    # def _mask_transform(self, mask):
    #     target = np.array(mask).astype('int32') - 1
    #     return torch.from_numpy(target).long()

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()-1




def _get_ade20k_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'val':
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        return train_img_paths + val_img_paths, train_mask_paths + val_mask_paths

    return img_paths, mask_paths

if __name__ == '__main__':
    tr_dataset= ADE20KSegmentation(root= '/mnt/venus-fast/segment_datasets/ADE20K')

    print(len(tr_dataset))
    img, segm= tr_dataset[25]
    print('img:', type(img), img.size)
    print('segm:', type(segm), segm.shape)

