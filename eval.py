import torch
import torch.nn.functional as F

import torch.nn as nn
from pytorch_msssim import ssim
from tqdm import tqdm


SSIM_WEIGHTS = [1, 10, 100, 1000]

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
loss_mse = nn.MSELoss(reduction='mean').cuda()


def psnr(original, contrast):
    mse = torch.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1 # 255.0
    PSNR = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return PSNR



def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = loader.batch_size#len(loader)  # the number of batch
    a = 0
    tot = 0
    tot1 = 0
    tot2 = 0
    tot3 = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, imgsr, gt = batch['image'], batch['mask'], batch['gt']
 
            if torch.cuda.is_available():
                
                # x0 = x0.cuda()
                gt = gt.cuda()
            with torch.no_grad():
                x,hphms,_,_,_,_,_ = net(imgs,imgsr) 

            x0 = x

            # print(psnr(x[0], gt))
            tot3 += psnr(x0, gt)
            tot2 += ssim(x0, gt)
            tot1 += mse_loss(x0, gt)

            pbar.update()
            a += 1
    net.train()
    print('psnr')
    print(tot3/a)
    print('ssim')
    print(tot2/a)
    return tot1 / n_val