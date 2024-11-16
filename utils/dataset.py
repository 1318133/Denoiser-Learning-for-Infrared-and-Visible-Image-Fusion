from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
# import gdaly
import tifffile as tiff
import torch.nn as nn
from torchvision.transforms import functional as F

def downsample(img): 
    return cv2.pyrDown( cv2.pyrDown(img) )

def upsample(img): 
    return cv2.pyrUp( cv2.pyrUp(img) )

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, gt, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.gt = gt
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    # def preprocess(cls, pil_img, scale):
    #     if len(pil_img.shape) == 2:
    #     #     w, h, chan = pil_img.shape
    #     #     newW, newH = int(scale * w), int(scale * h)
    #     #     assert newW > 0 and newH > 0, 'Scale is too small'
    #     #     # pil_img = pil_img.resize((newW, newH, c))
    #     #     pil_img = F.resize(pil_img,(newW, newH, chan))
    #     # else:
    #     #     w, h = pil_img.shape
    #     #     newW, newH = int(scale * w), int(scale * h)
    #     #     assert newW > 0 and newH > 0, 'Scale is too small'
    #     #     # pil_img = pil_img.resize((newW, newH, c))
    #     #     pil_img = F.resize(pil_img,(newW, newH))
    #         pil_img = np.expand_dims(pil_img, axis=2)

    #     # img_nd = np.array(pil_img)

    #     # if len(img_nd.shape) == 2:
    #     #     img_nd = np.expand_dims(img_nd, axis=2)

    #     # HWC to CHW
    #     w, h, chan = pil_img.shape
    #     img_trans = pil_img.reshape(chan, w, h)# img_nd.transpose((2, 0, 1))
    #     if img_trans.max() > 1:
    #         # img_trans = img_trans / 255
    #         img_trans = (img_trans-img_trans.min()) / (img_trans.max()-img_trans.min())
    #     # print(1)
    #     return img_trans

    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        gt_file = glob(self.gt + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        # # dt_gt = gdal.Open(gt)
        # # dt_lr = gdal.Open(lr)
        # # dt_pan = gdal.Open(pan)
            
        # # img_gt = dt_gt.ReadAsArray() # (c, h, w)
        # # img_lr = dt_lr.ReadAsArray()
        # # img_pan = dt_pan.ReadAsArray() 
        # mask = tiff.imread(mask_file[0])#.astype(np.float32)
        # # mask = torch.tensor(mask, dtype=torch.float)
        # # m = nn.Upsample(scale_factor=4, mode='nearest')
        # # mask = m(mask)
        # # mask = self.preprocess(mask, 4)
        # mask = upsample(mask)
        # img =  tiff.imread(img_file[0])
        # # img = img.unsqueeze(-1)
        # gt =  tiff.imread(gt_file[0])
        # img3 = downsample(img)
        # img3 = upsample(img3)

        mask = Image.open(mask_file[0])
        # mask = self.preprocess(mask, 4)
        img = Image.open(img_file[0])
        gt = Image.open(gt_file[0])

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        
        img2 = self.preprocess(img, 0.5)
        mask2 = self.preprocess(mask, 0.5)

        img3 = self.preprocess(img, self.scale)
        # img3 = self.preprocess(img, 0.25)
        mask3 = self.preprocess(mask, 0.25)

        img4 = self.preprocess(img, 0.125)
        mask4 = self.preprocess(mask, 0.125)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        gt = self.preprocess(gt, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'gt': torch.from_numpy(gt).type(torch.FloatTensor),
            'image2': torch.from_numpy(img2).type(torch.FloatTensor),
            'mask2': torch.from_numpy(mask2).type(torch.FloatTensor),
            'image3': torch.from_numpy(img3).type(torch.FloatTensor),
            'mask3': torch.from_numpy(mask3).type(torch.FloatTensor),
            'image4': torch.from_numpy(img4).type(torch.FloatTensor),
            'mask4': torch.from_numpy(mask4).type(torch.FloatTensor),
        }


class TestDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        # l = listdir(imgs_dir)
        # for i in range(len(l)):  
        #     l[i] = l[i].split('.')  
        #     l[i][0] = int(l[i][0])  

        # l.sort()  

        # for i in range(len(l)):  
        #     l[i][0] = str(l[i][0])  
        #     l[i] = l[i][0] + '.' + l[i][1]  

        # m = listdir(masks_dir)
        # for i in range(len(m)):  
        #     m[i] = m[i].split('.')  
        #     m[i][0] = int(m[i][0])  

        # m.sort()  

        # for i in range(len(m)):  
        #     m[i][0] = str(m[i][0])  
        #     m[i] = m[i][0] + '.' + m[i][1]  
        self.imlist = listdir(imgs_dir)
        self.imlist.sort(key=lambda x:int(x[:-4])) # self.imlist.sort(key=lambda x:int(x[11:-4]))
        self.malist = listdir(masks_dir)
        self.malist.sort(key=lambda x:int(x[:-4]))
        self.ids = [splitext(file)[0] for file in self.imlist#listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.mds = [splitext(file)[0] for file in self.malist#listdir(masks_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.mds)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mdx = self.mds[i]
        mask_file = glob(self.masks_dir + mdx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')


        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {mdx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = cv2.imread(mask_file[0])
        # img = cv2.imread(img_file[0])
        # mask = tiff.imread(mask_file[0])#.astype(np.float32)
        # mask = upsample(mask)
        # img =  tiff.imread(img_file[0])
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
