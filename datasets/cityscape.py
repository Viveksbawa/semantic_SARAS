from torch.utils.data import Dataset
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
#import os.path as path
import torch
import glob
import os
import numpy as np



def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, -1)
        # The values above are remapped to the following
        self.new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                    8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)
        
        self.color_encoding = OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('road', (128, 64, 128)),
                ('sidewalk', (244, 35, 232)),
                ('building', (70, 70, 70)),
                ('wall', (102, 102, 156)),
                ('fence', (190, 153, 153)),
                ('pole', (153, 153, 153)),
                ('traffic_light', (250, 170, 30)),
                ('traffic_sign', (220, 220, 0)),
                ('vegetation', (107, 142, 35)),
                ('terrain', (152, 251, 152)),
                ('sky', (70, 130, 180)),
                ('person', (220, 20, 60)),
                ('rider', (255, 0, 0)),
                ('car', (0, 0, 142)),
                ('truck', (0, 0, 70)),
                ('bus', (0, 60, 100)),
                ('train', (0, 80, 100)),
                ('motorcycle', (0, 0, 230)),
                ('bicycle', (119, 11, 32))
        ])

        self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        
        self.train= ['/leftImg8bit_trainvaltest/leftImg8bit/train', '/gtFine_trainvaltest/gtFine/train']
        self.val= ['/leftImg8bit_trainvaltest/leftImg8bit/val', '/gtFine_trainvaltest/gtFine/val']
        self.test= ["/leftImg8bit_trainvaltest/leftImg8bit/test", "/gtFine_trainvaltest/gtFine/test"]

    @staticmethod
    def remap(image, old_values, new_values):
        assert isinstance(image, Image.Image) or isinstance(
            image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
        assert type(new_values) is tuple, "new_values must be of type tuple"
        assert type(old_values) is tuple, "old_values must be of type tuple"
        assert len(new_values) == len(
            old_values), "new_values and old_values must have the same length"

        # If image is a PIL.Image convert it to a numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Replace old values by the new ones
        tmp = np.zeros_like(image)
        for old, new in zip(old_values, new_values):
            # Since tmp is already initialized as zeros we can skip new values
            # equal to 0
            if new != 0:
                tmp[image == old] = new

        return Image.fromarray(tmp)
    
    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return int(((x - 1) // p + 1) * p)


#--------------------------------------------------------------------------

class TrainDataset(BaseDataset):
    def __init__(self, opt, batch_per_gpu=5, **kwargs):
        super(TrainDataset, self).__init__(**kwargs)
        self.root_dataset = opt.root_dataset
        self.batch_per_gpu = batch_per_gpu
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.padding_constant = opt.padding_constant
        self.imgSizes = opt.imgSizes

        self.cur_idx = 0
        self.img_paths= self.read_train_files()
        self.colors= np.array(list(self.color_encoding.values()), dtype=np.uint8)    
    
    def read_train_files(self):
        imgEnd='leftImg8bit.png'
        labelEnd= 'gtFine_labelIds.png'
        
        cities= glob.glob(self.root_dataset+ self.train[0]+'/*')
        data=[]
        for city in cities:
            for img in glob.glob(city+'/*'):
                label= img.replace(self.train[0], self.train[1])
                label= label.replace(imgEnd, labelEnd)
                if os.path.exists(label) and os.path.exists(img):
                    data.append((label, img))
                else:
                    print('Can not find the label:', label)
        return(data)

    def __len__(self):
        return(100000)


    def __getitem__(self, index):
        if self.cur_idx==0 or (self.cur_idx+self.batch_per_gpu >= len(self.img_paths)):
            np.random.shuffle(self.img_paths)
            self.cur_idx=0
        
        short_side= np.random.choice(self.imgSizes)
        short_side = self.round2nearest_multiple(short_side, self.padding_constant)

        list_img= []
        list_segm= []
        for i in range(self.batch_per_gpu):
            segm_path, img_path=  self.img_paths[self.cur_idx]
            self.cur_idx += 1
        
            img= Image.open(img_path)
            label= Image.open(segm_path)
            segm= self.remap(label, self.full_classes, self.new_classes)

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            
            img= transforms.Resize(short_side, Image.BILINEAR)(img)
            segm= transforms.Resize(short_side // self.segm_downsampling_rate, 
                        Image.NEAREST)(segm)

            image= self.img_transform(img)
            segm_map= self.segm_transform(segm)
            # print(image.shape, segm_map.shape)
            list_img.append(image)
            list_segm.append(segm_map)

        return self.make_tensor(list_img, list_segm, short_side)


    def make_tensor(self, img_list, segm_list, short_size):
        max_wh = [0,0]
        for seg in img_list:
            if seg.shape[1] > max_wh[0]: max_wh[0] = seg.shape[1]
            if seg.shape[2] > max_wh[1]: max_wh[1] = seg.shape[2]
        max_wh[0] = self.round2nearest_multiple(max_wh[0], self.padding_constant)
        max_wh[1] = self.round2nearest_multiple(max_wh[1], self.padding_constant)
        # print(max_wh)

        batch_img= torch.zeros(self.batch_per_gpu, 3, max_wh[0], max_wh[1])
        batch_segm= torch.zeros(self.batch_per_gpu, 
                    max_wh[0] // self.segm_downsampling_rate, 
                    max_wh[1] // self.segm_downsampling_rate).long()
        # print(batch_img.shape, batch_segm.shape)

        for i, (img, segm) in enumerate(zip(img_list,segm_list)):
            batch_img[i, :, :img.shape[1], :img.shape[2]] = img
            batch_segm[i, :segm.shape[0], :segm.shape[1]] = segm

        return batch_img, batch_segm



class ValDataset(BaseDataset):
    def __init__(self, opt, **kwargs):
        super(ValDataset, self).__init__(**kwargs)
        self.root_dataset = opt.root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.padding_constant = opt.padding_constant
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize

        self.cur_idx = 0
        self.img_paths= self.read_val_files()
        self.colors= np.array(list(self.color_encoding.values()), dtype=np.uint8)    
    
    def read_val_files(self):
        
        imgEnd='leftImg8bit.png'
        labelEnd= 'gtFine_labelIds.png'
        cities= glob.glob(self.root_dataset+ self.val[0]+'/*')
        data=[]
        for city in cities:
            img_list= glob.glob(city+'/*')
            for img in img_list:
                label= img.replace(self.val[0], self.val[1])
                label= label.replace(imgEnd, labelEnd)
                if os.path.exists(label) and os.path.exists(img):
                    data.append((label, img))
                else:
                    print('Can not find the label:', label)
        return(data)

    def __len__(self):
        return(len(self.img_paths))


    def __getitem__(self, index):
        segm_path, img_path=  self.img_paths[index]

        img= Image.open(img_path)
        label= Image.open(segm_path)
        segm= self.remap(label, self.full_classes, self.new_classes)

        img_list = []
        for short_size in self.imgSizes:
            round_size= self.round2nearest_multiple(short_size, self.padding_constant)
            img_resized= transforms.Resize(round_size, Image.BILINEAR)(img)
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        # round_size = self.round2nearest_multiple(max(self.imgSizes), self.padding_constant)
        segm_resized= transforms.Resize(round_size, Image.NEAREST)(segm)
        segm_resized = self.segm_transform(segm_resized)
        batch_segms = torch.unsqueeze(segm_resized, 0)

        return img_list, batch_segms



if __name__ == '__main__':
    from config import cfg
    cfg.merge_from_file("config/city-resnet50upernet.yaml")
    tr_data= TrainDataset(opt= cfg.DATASET, batch_per_gpu=4)
    # print('Number of examples', len(tr_data.img_paths))
    img, seg = tr_data[10]
    print(img.shape, seg.shape)

    # print(tr_data.img_paths)
    # print(img.shape, label.shape)
    # print(img[0].shape, '\n', img[1].shape, '\n', img[2].shape, label.shape)
    # print(img[2].shape, label[2].shape)
    # print(img[3].shape, label[3].shape)
    # print(img[4].shape, label[4].shape)
