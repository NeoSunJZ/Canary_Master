import torch
import torch.nn as nn
from collections import OrderedDict


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """

        Args:
            in_channels: the channels of input
            out_channels: the channels of output
            use_1x1conv: whether use the 1*1 convolution
            stride:
        """
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Define 1*1 convolution
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv3 = None

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.bn3(self.conv3(x))

        return self.relu(y + x)


def residual_block(in_channels, out_channels, residual_number, first_residual_block=False):

    residual_block_list = []
    for i in range(residual_number):
        if i == 0 and not first_residual_block:
            residual_block_list.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            residual_block_list.append(Residual(out_channels, out_channels))

    return residual_block_list


class ResNet19(nn.Module):
    def __init__(self):
        super(ResNet19, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3, bias=False)),  # 64*28*28
            ('Bn1', nn.BatchNorm2d(64)),
            ('Relu1', nn.ReLU()),
            ('Pool1', nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))
        ]))
        self.c2 = nn.Sequential(*residual_block(in_channels=64, out_channels=64, residual_number=2, first_residual_block=True))
        self.c3 = nn.Sequential(*residual_block(in_channels=64, out_channels=128, residual_number=2))
        self.c4 = nn.Sequential(*residual_block(in_channels=128, out_channels=256, residual_number=2))
        self.c5 = nn.Sequential(*residual_block(in_channels=256, out_channels=512, residual_number=2))
        self.c6 = nn.Sequential(OrderedDict([
            # ('Pool6', nn.AdaptiveAvgPool2d((1, 1))),
            ('FullCon6', nn.Linear(in_features=512 * 1 * 1, out_features=1000)),
            ('Sig16', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)

        output = torch.flatten(output, 1)
        output = self.c6(output)

        return output


class ResNet19Light(nn.Module):
    def __init__(self):
        super(ResNet19Light, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=3, bias=False)),
            ('Bn1', nn.BatchNorm2d(64)),
            ('Relu1', nn.ReLU()),
            ('Pool1', nn.MaxPool2d(kernel_size=(2, 2), padding=1))
        ]))
        self.c2 = nn.Sequential(*residual_block(in_channels=64, out_channels=64, residual_number=2, first_residual_block=True))
        self.c3 = nn.Sequential(*residual_block(in_channels=64, out_channels=128, residual_number=2))
        self.c4 = nn.Sequential(*residual_block(in_channels=128, out_channels=256, residual_number=2))
        self.c5 = nn.Sequential(*residual_block(in_channels=256, out_channels=512, residual_number=2))
        self.c6 = nn.Sequential(OrderedDict([
            ('FullCon6', nn.Linear(in_features=512 * 3 * 3, out_features=1024)),
            ('Relu6', nn.ReLU()),
        ]))
        self.c7 = nn.Sequential(OrderedDict([
            ('FullCon7', nn.Linear(in_features=1024, out_features=10)),
            ('Sig7', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)

        output = torch.flatten(output, 1)
        output = self.c6(output)
        output = self.c7(output)

        return output









