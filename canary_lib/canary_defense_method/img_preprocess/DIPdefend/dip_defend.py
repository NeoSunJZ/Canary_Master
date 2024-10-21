from torchvision import models
import os

import numpy as np
from .models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from .utils.denoising_utils import *
import torch
from torch.autograd import Variable
import argparse

# parser = argparse.ArgumentParser(description='Args for DIPDefend')
# parser.add_argument('--fname',
#                     help='The name of the attacked image', type=str)
# parser.add_argument('--out_dir', default='.',
#                     help='The directory used to save the output', type=str)
# parser.add_argument('--num_iter', default=4000, type=int,
#                     help='Number of total iterations to run')
# parser.add_argument('--input_depth', default=1, type=int,
#                     help='Input depth for the generator')
# parser.add_argument('--lr', default=0.01, type=float,
#                     help='Learning rate for the optimizer')
# parser.add_argument('--Lambda', default=0.002, type=float,
#                     help='Hyperparameter of SES')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """

    im_as_ten = torch.from_numpy(cv2im).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=False).to(0)
    return im_as_var


def adapative_psnr(img1, img2, size=32):
    psnr, area_cnt = [], 0
    _, h, w = img1.shape

    for i in range(int(h // size)):
        for j in range(int(w // size)):
            img1_part = img1[:, i * size:(i + 1) * size, j * size:(j + 1) * size]
            img2_part = img2[:, i * size:(i + 1) * size, j * size:(j + 1) * size]
            psnr.append(compare_psnr(img1_part, img2_part))
            area_cnt += 1
    psnr = np.array(psnr).min()
    return psnr


# args = parser.parse_args()

# attack_fname = args.fname
# img_noisy_pil = crop_image(get_image(attack_fname, -1)[0], d=32)
#
# img_noisy_np = pil_to_np(img_noisy_pil)

# out_dir = args.out_dir

# LAMBDA = args.Lambda
# reg_noise_std = -1. / 20.
# LR = args.lr
# num_iter = args.num_iter
# input_depth = args.input_depth
LAMBDA = 0.002
reg_noise_std = -1. / 20.
LR = 0.01
num_iter = 400
input_depth = 1
series, out_series, delta_series = [], [], []
ii = False


def Net():
    return skip(input_depth, 3,
                num_channels_down=[4, 8, 16],
                num_channels_up=[4, 8, 16],
                num_channels_skip=[0, 0, 0],
                upsample_mode='bilinear').type(dtype)



def closure(net, net_input, psnr_max_img, SES_img, img_noisy_torch, net_input_saved, noise, mse, img_noisy_np):

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse(out.mean(dim=0, keepdim=True), img_noisy_torch)
    total_loss.backward()

    psrn_gt = adapative_psnr(img_noisy_np, out.detach().cpu().numpy().mean(axis=0))
    ii=False
    if len(series) == 0:
        series.append(psrn_gt)
        out_series.append(psrn_gt)
    elif len(series) == 1:
        series.append(psrn_gt)
        delta_series.append(series[1] - series[0])
        out_series.append(LAMBDA * series[-1] + (1 - LAMBDA) * (out_series[-1] + delta_series[-1]))
    else:
        series.append(psrn_gt)
        s = LAMBDA * series[-1] + (1 - LAMBDA) * (out_series[-1] + delta_series[-1])
        t = LAMBDA * (s - out_series[-1]) + (1 - LAMBDA) * (delta_series[-1])
        out_series.append(s);
        delta_series.append(t)
        if out_series[-1] > np.array(out_series[:-1]).max():
            ii=True
            SES_img = out.detach().cpu().numpy().mean(axis=0)
        # if series[-1] > np.array(series[:-1]).max():
        #     psnr_max_img = out.detach().cpu().numpy().mean(axis=0)

    print(ii)
    return total_loss, SES_img



# optimize('adam', p, closure, LR, num_iter)

def dip_defend(img):
    img_noisy_np = img.detach().cpu().numpy()
    net = Net()
    psnr_max_img = None
    SES_img = None

    net_inputs = []
    for i in range(1):
        net_input = get_noise(input_depth, 'noise', (32, 32)).type(dtype).detach()
        net_inputs.append(net_input.squeeze(0).cpu().numpy())
    net_input = torch.FloatTensor(np.array(net_inputs)).type(dtype).detach().to(0)

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params('net,input', net, net_input)
    optimizer = torch.optim.Adam(p, lr=LR)

    for j in range(num_iter):
        optimizer.zero_grad()
        total_loss, SES_img = closure(net, net_input, psnr_max_img, SES_img, img_noisy_torch, net_input_saved, noise, mse, img_noisy_np)
        optimizer.step()
    return SES_img
# np.save(os.path.join(out_dir, 'defense_inflection.npy'), SES_img)
# np_to_pil(SES_img).save(os.path.join(out_dir, 'defense_inflection.png'))
