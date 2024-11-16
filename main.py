import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim, sigmoid, zeros_like
from tqdm import tqdm
from imageio import imwrite
from eval import eval_net

import torch.nn.functional as F
from vgg import Vgg16
from vgg4out import Vgg16_4
from PIL import Image
# import tifffile as tiff

from unet import viir,viir_test

from pytorch_msssim import ssim
import cv2

# from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, TestDataset
from torch.utils.data import DataLoader, random_split

dir_img = '/home/s1u1/dataset/MSRS-main/train/ir_1ch_128/'
dir_mask = '/home/s1u1/dataset/MSRS-main/train/vi_1ch_128/'


gt = '/home/s1u1/dataset/MSRS-main/train/ir_1ch_128/'


dir_checkpoint = 'checkpoints/'
showpathimg = 'epoch_fuseimg_show/img/'
showpathvis = 'epoch_fuseimg_show/vis/'
showpathinf = 'epoch_fuseimg_show/inf/'
showpathadd = 'epoch_fuseimg_show/add/'
showpathxo = 'epoch_fuseimg_show/xo/'
showpathyo = 'epoch_fuseimg_show/yo/'
showpathfxo = 'epoch_fuseimg_show/fxo/'
showpathfyo = 'epoch_fuseimg_show/fyo/'

SSIM_WEIGHTS = [1, 10, 100, 1000]

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
loss_mse = nn.MSELoss(reduction='mean').cuda()
sig = nn.Sigmoid()

class grad(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, inchannel =3):
        super().__init__()
        kernel = torch.Tensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        self.conv = nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, padding_mode='replicate')
        
        self.conv.weight.data = kernel     

    def forward(self, x):
        return self.conv(x)

def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    # features = features.unsqueeze(1)
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads

def imgshow(img, showpath, index):
    img = img[1,:,:,:]
    img_final = img.detach().cpu().numpy()
    img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
    img = img_final.transpose(1,2,0)
    img = img.astype('uint8')
    if img.shape[2] == 1:
        img = img.reshape([img.shape[0], img.shape[1]])
    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, img)
    return img

dtype = torch.cuda.FloatTensor   
vgg4 = Vgg16_4().type(dtype)

def vgg_gradmeasure_loss(out,img,imgs):
    a1,a2,a3,a4 = vgg4(img)
    b1,b2,b3,b4 = vgg4(imgs)
    c1,c2,c3,c4 = vgg4(out)
    ag1 = torch.abs(features_grad(a1))
    ag2 = torch.abs(features_grad(a2))
    ag3 = torch.abs(features_grad(a3))
    ag4 = torch.abs(features_grad(a4))
    bg1 = torch.abs(features_grad(b1))
    bg2 = torch.abs(features_grad(b2))
    bg3 = torch.abs(features_grad(b3))
    bg4 = torch.abs(features_grad(b4))

    am1 = (ag1.mean())/(ag1.mean()+bg1.mean()+0.0000001)
    bm1 = 1-am1
    am2 = (ag2.mean())/(ag2.mean()+bg2.mean()+0.0000001)
    bm2 = 1-am2
    am3 = (ag3.mean())/(ag3.mean()+bg3.mean()+0.0000001)
    bm3 = 1-am3
    am4 = (ag4.mean())/(ag4.mean()+bg4.mean()+0.0000001)
    bm4 = 1-am4
    loss1 = am1*mae_loss(a1,c1)+bm1*mae_loss(b1,c1)
    loss2 = am2*mae_loss(a2,c2)+bm2*mae_loss(b2,c2)
    loss3 = am3*mae_loss(a3,c3)+bm3*mae_loss(b3,c3)
    loss4 = am4*mae_loss(a4,c4)+bm4*mae_loss(b4,c4)

    loss = loss1 + loss2 + loss3 + loss4 

    return loss

def base(fxout, fyout, img, imgs, true_masks):
    weight1 = 1
    weight2 = 1
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)

    loss = 20 * loss_2 + loss_1 
    return loss, imgs, true_masks

def train_net(net, 
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.25):
    ph = 1
    al = 0 #低于此全部置0
    c = 3500

    dataset = BasicDataset(dir_img, dir_mask, gt, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train, val = random_split(dataset, [n_train, n_val])
    # n_val9 = int(n_val*0.5)
    # n_val1 = n_val-n_val9
    # val2, val = random_split(val1, [n_val9, n_val1])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    index = 1
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # load vgg network
    dtype = torch.cuda.FloatTensor

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)

    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                gth = batch['gt']
                imgs2 = batch['image2']
                true_masks2 = batch['mask2']
                imgs3 = batch['image3']
                true_masks3 = batch['mask3']
                imgs4 = batch['image4']
                true_masks4 = batch['mask4']
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    gth = gth.cuda()
                    imgs2 = imgs2.cuda()
                    true_masks2 = true_masks2.cuda()
                    imgs3 = imgs3.cuda()
                    true_masks3 = true_masks3.cuda()
                    imgs4 = imgs4.cuda()
                    true_masks4 = true_masks4.cuda()

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                gth = gth.to(device=device, dtype=torch.float32)
                imgs2 = imgs2.to(device=device, dtype=torch.float32)
                true_masks2 = true_masks2.to(device=device, dtype=torch.float32)
                imgs3 = imgs3.to(device=device, dtype=torch.float32)
                true_masks3 = true_masks3.to(device=device, dtype=torch.float32)
                imgs4 = imgs4.to(device=device, dtype=torch.float32)
                true_masks4 = true_masks4.to(device=device, dtype=torch.float32)

                enhance_image_1,enhance_image,r,out,img,nx,ny = net(imgs,true_masks)

                loss_base,_,_ = base(out, img, enhance_image_1, imgs, true_masks)

                loss_vgg = vgg_gradmeasure_loss(enhance_image_1,imgs,true_masks)

                lossall = loss_base+loss_vgg +10*enhance_image+10*r#loss_vgg + #  + loss_base           #loss1 #+ 100*gi_loss# + 10*loss_col# + loss_fu # lossRGB + 

                pbar.set_postfix(**{'loss1 (batch)': loss_base.item(),'loss2 (batch)': loss_vgg.item(),'loss3 (batch)': enhance_image.item(),'loss4 (batch)': r.item()})#,'loss3 (batch)': loss3.item(),'loss4 (batch)': loss4.item(),'loss3 (batch)': loss3.item(),'loss3 (batch)': loss3.item()})
                optimizer.zero_grad()
                # loss2.backward(retain_graph=True)
                # loss1.backward()
                # loss3.backward()
                lossall.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                ######
                if global_step == ((n_train // batch_size)*index):
                    g = imgshow(enhance_image_1,showpathxo,index)
                    print(optimizer.state_dict()['param_groups'][0]['lr'])
#################
                    index += 1  
                #####
                if global_step % (n_train // (1  * batch_size)) == 0:
                # if global_step == 5:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)

                    if net.n_classes > 1:
                        logging.info('Validation mse: {}'.format(val_score))
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=75.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', type=str, default=False,
                        help='If test images turn True, train images turn False')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    pthf = None#'/home/s1u1/code/overall/deno_fusion/checkpointsall/075_MSRStrian_oldnet_lossvggc_nonssim_CP_epoch29.pth'#'/home/s1u1/code/overall/pansharpning_mtask/checkpointsall/075_lrdown_roadscene_mse_ssim_vgg_CP_epoch23.pth'


    vi = '/home/s1u1/dataset/MSRS-main/detection/vi20_1ch/'
    ir = '/home/s1u1/dataset/MSRS-main/detection/ir20_1ch/'

    path = '/home/s1u1/code/overall/deno_fusion/075_MSRStrian_oldnet_lossvggc_nonssim_msrs_CP_epoch29/'#'/home/s1u1/dataset/MSRS-main/detection/deno_trainbymsrslargev2_1ch/'#tno_trainbymsrs/#'./outputs/'#'/home/s1u1/code/overall/pansharpning_mtask/detect_train/'#/home/s1u1/dataset/ConvertedImages/deno_trainbymsrslarge_1ch/'
    pathadd = './outputsadd/'
    # path1 = './ir_outputs/'
    # path2 = './vi_outputs/'

    dataset = TestDataset(ir, vi)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    net = viir.wtNet(n_channels=3, n_classes=1, bilinear=True, pthfile=pthf)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    index = 1
    Time = 0
    if args.test:
        for im in test_loader:
            ir = im['image']
            vi = im['mask']
            if torch.cuda.is_available():
                ir = ir.cuda()
                vi = vi.cuda()
            # Net = Wdenet.wtNet(1, 1)
            # Net = unet_model.UNet(1,1)
            Net = viir_test.wtNet(3, 3, pthfile=pthf)
            Net = Net.cuda()
            # ##########################
            add = ir*0.5 + vi*0.5
            img_final = add.detach().cpu().numpy()
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(0, 2, 3, 1)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png'
            path_out = pathadd + file_name
            # index += 1            
            imwrite(path_out, img)
            # ##########################
            start = time.time()
            # img, _, _ = Net(vi, ir)
            img,_,_,_,_,_,_ = Net(ir, vi)
            # img = Net(vi, ir)
            img_final = img.detach().cpu().numpy() 
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(0, 2, 3, 1)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            end = time.time()
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png'
            path_out = path + file_name
            # save_img(img.cpu().data, path_out, mode='CMYK')
            # index += 1            
            imwrite(path_out, img)
            Time += end-start
            print(index)
            print(end-start)
            index += 1  
        average_time = Time/(len(test_loader))  
        print(average_time) 

    else:
        try:
            train_net(net=net,
                    # net1=net1,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
