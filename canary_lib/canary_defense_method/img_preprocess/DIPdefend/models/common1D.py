import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module


class GaussConv1d(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, filter_size = 3, bias=True, var_w=1., var_b=1.):
        super(GaussConv1d, self).__init__()
        self.weight = Parameter(torch.Tensor(num_output_channels, num_input_channels, filter_size))
        # kaiming normal intialization
        stdv = math.sqrt(2 * var_w) / math.sqrt(num_input_channels * filter_size)
        self.weight.data.normal_(0, stdv)
        # nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if bias:
            self.bias = Parameter(torch.Tensor(num_output_channels))
            self.bias.data.normal_(0, math.sqrt(var_b))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias)


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes = [x.shape[2] for x in inputs]

        if np.all(np.array(inputs_shapes) == min(inputs_shapes)):

            inputs_ = inputs
        else:
            target_shape = min(inputs_shapes)

            inputs_ = []
            for inp in inputs: 
                diff = (inp.size(2) - target_shape) // 2 
                inputs_.append(inp[:, :, diff: diff + target_shape])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x;


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
            #return nn.ReLU();
        elif act_fun == 'LeakyReLU2':
            return nn.LeakyReLU(0.5, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    #return nn.BatchNorm1d(num_features)
    return Identity()


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool1d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool1d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)

    if pad == 'reflection':
        padder = nn.ReflectionPad1d(to_pad)
        to_pad = 0

    # Uniform initialization
    convolver = nn.Conv1d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    # kaiming_norm initialization
    # convolver = GaussConv1d(in_f, out_f, kernel_size, bias = bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
