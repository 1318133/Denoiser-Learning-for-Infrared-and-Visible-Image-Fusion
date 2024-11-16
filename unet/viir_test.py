import torch.nn.functional as F
import torch.nn as nn
# from unet.unet_parts import *
import numpy
import torch

from .denoising_diffusion_pytorch import *


class wtNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pthfile = 'F:/3line/checkpoints/CP_epoch13.pth'):
        super(wtNet, self).__init__()
         
        # net= UNet(n_channels=1, n_classes=1, bilinear=True)


        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 

        number_f = 32


        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = 128,
            timesteps = 1000,   # number of steps
            loss_type = 'l1',   # L1 or L2
            objective= 'pred_x0'
        )

        self.model1 = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8)
        )

        self.diffusion1 = GaussianDiffusion(
            self.model1,
            image_size = 128,
            timesteps = 1000,   # number of steps
            loss_type = 'l1',   # L1 or L2
            objective= 'pred_x0'
        )
        

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(2,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv7 = nn.Conv2d(number_f*2,1,3,1,1,bias=True,padding_mode='reflect') 
        self.sig = nn.Sigmoid()

        if pthfile is not None:
            # self.load_state_dict(torch.save(torch.load(pthfile), pthfile,_use_new_zipfile_serialization=False), strict = False)  # 训练所有数据后，保存网络的参数
            self.load_state_dict(torch.load(pthfile), strict = False)
        


    def forward(self, x, y):#(self, x, y):# x = RGB; y = NIR


        c = torch.cat((y,x),1)
        x1 = self.relu(self.conv1(c))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.conv6(torch.cat([x2,x5],1)))
        out = F.tanh(self.conv7(torch.cat([x1,x6],1)))



        return out,out,out,out,out,out,out # enhance_image_1(x0),enhance_image(y0),r(vis),out(img),img(inf)

