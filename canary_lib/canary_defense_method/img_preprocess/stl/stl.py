from __future__ import print_function
from builtins import input
from builtins import range
import pyfftw
import os
import numpy as np
import cv2
import sys
import sporco.metric as sm
from sporco.admm import cbpdn
from sporco import util
import sporco
import multiprocessing
from multiprocessing import Pool
import torch


class STL():
    def __init__(self, device, dataset, npy_name):
        npy_name_split = npy_name.split("_")
        # K = int(npy_name_split[0]) # filter number, 64
        # PATCH_SIZE = int(npy_name_split[1][1:]) # filter size, 8
        self.lmbda = float(npy_name_split[2][2:])  # sparse coefficient, 0.1 to 0.3 is good, we usually use #0.1
        # self.P = PATCH_SIZE
        self.npd = 16
        self.fltlmbd = 5
        dic_path = os.path.join('/home/zhangda/projects/Canary_Master/canary_lib/canary_defense_method/img_preprocess/stl/stl_basis', dataset, npy_name + '.npy')
        self.basis = np.load(dic_path)
        self.dataset = dataset
        self.opt = cbpdn.ConvBPDN.Options(
            {'Verbose': False, 'MaxMainIter': 200, 'RelStopTol': 5e-3, 'AuxVarObj': False})
        self.device = device

    def forward_single_img(self, im):
        sl, sh = sporco.signal.tikhonov_filter(im, self.fltlmbd, self.npd)
        b = cbpdn.ConvBPDN(self.basis, sh, self.lmbda, self.opt)
        X = b.solve()
        shr = b.reconstruct().squeeze()
        imgr = sl + shr
        imgr = imgr / imgr.max()
        return imgr

    def forward(self, x):
        tmp_x = x.detach().clone().cpu().numpy()
        # lst_img = []
        # for img in tmp_x:
        #     img_out = self.forward_single_img(img)
        #     img_out = torch.from_numpy(img_out).permute(2, 0, 1)
        #     lst_img.append(img_out)
        #
        # output = x.new_tensor(torch.stack(lst_img)).to(self.device)
        img = self.forward_single_img(tmp_x)
        output = torch.from_numpy(img)*255
        # output = torch.from_numpy(img).permute(2, 0, 1)
        return output


