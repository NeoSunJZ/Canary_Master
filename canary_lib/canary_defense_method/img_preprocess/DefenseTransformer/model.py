from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np


 
# From the supplementary of STN paper. conv[32,5,1,2]-max[2]-conv[32,5,1,2]-fc[32]-fc[32].
class Single_ST(nn.Module):
    def __init__(self):
        super(Single_ST, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.contiguous().view(-1, 32 * 16 * 16)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # print(theta)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        return x



class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class UNet(nn.Module):
    def __init__(self, h_in, w_in, block, fwd_in, fwd_out, num_fwd, back_out, num_back, use_Single_ST):
        super(UNet, self).__init__()

        h, w = [], []
        for i in range(len(num_fwd)):
            h.append(h_in)
            w.append(w_in)
            h_in = int(np.ceil(float(h_in) / 2))
            w_in = int(np.ceil(float(w_in) / 2))

        # if block is Bottleneck:
        #     expansion = 4
        # else:
        expansion = 1
        
        fwd = []
        n_in = fwd_in
        for i in range(len(num_fwd)):
            group = []
            for j in range(num_fwd[i]):
                if j == 0:
                    if i == 0:
                        stride = 1
                    else:
                        stride = 2
                    group.append(block(n_in, fwd_out[i], stride = stride))
                else:
                    group.append(block(fwd_out[i] * expansion, fwd_out[i]))
            n_in = fwd_out[i] * expansion
            fwd.append(nn.Sequential(*group))
        self.fwd = nn.ModuleList(fwd)

        upsample = []
        back = []
        n_in = (fwd_out[-2] + fwd_out[-1]) * expansion
        for i in range(len(num_back) - 1, -1, -1):
            upsample.insert(0, nn.Upsample(size = (h[i], w[i]), mode = 'bilinear'))
            group = []
            for j in range(num_back[i]):
                if j == 0:
                    group.append(block(n_in, back_out[i]))
                else:
                    group.append(block(back_out[i] * expansion, back_out[i]))
            if i != 0:
                n_in = (back_out[i] + fwd_out[i - 1]) * expansion
            back.insert(0, nn.Sequential(*group))
        self.upsample = nn.ModuleList(upsample)
        self.back = nn.ModuleList(back)

        self.final = nn.Conv2d(back_out[0] * expansion, fwd_in, kernel_size = 1, bias = False)

        self.use_Single_ST = use_Single_ST
        if self.use_Single_ST:
            self.singleST = Single_ST()

    def forward(self, x):
        out = x
        outputs = []
        for i in range(len(self.fwd)):
            out = self.fwd[i](out)
            if i != len(self.fwd) - 1:
                outputs.append(out)
        
        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out)
            out = torch.cat((out, outputs[i]), 1)
            out = self.back[i](out)
        out = self.final(out)
        out += x

        if self.use_Single_ST:
            out = self.singleST(out)
        return out
