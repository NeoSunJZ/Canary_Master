import torch
import torch.nn as nn
from collections import OrderedDict


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4)),
            ('Relu1', nn.ReLU()),
            ('LRN1', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('Pool1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ('Conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), groups=2)),
            ('Relu2', nn.ReLU()),
            ('LRN2', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('Pool2', nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        ]))
        self.c3 = nn.Sequential(OrderedDict([
            ('Conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1)),
            ('Relu3', nn.ReLU()),
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ('Conv4', nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1, groups=2)),
            ('Relu4', nn.ReLU()),
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ('Conv5', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1, groups=2)),
            ('Relu5', nn.ReLU()),
            ('Pool5', nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        ]))
        self.c6 = nn.Sequential(OrderedDict([
            ('FullCon6', nn.Linear(in_features=256*6*6, out_features=4096)),
            ('Relu6', nn.ReLU()),
            ('Drop6', nn.Dropout(p=0.5))
        ]))
        self.c7 = nn.Sequential(OrderedDict([
            ('FullCon7', nn.Linear(in_features=4096, out_features=4096)),
            ('Relu7', nn.ReLU()),
            ('Drop7', nn.Dropout(p=0.5))
        ]))
        self.c8 = nn.Sequential(OrderedDict([
            ('FullCon8', nn.Linear(in_features=4096, out_features=1000)),
            ('Sig8', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)

        output = output.view(-1, 256)
        output = self.c6(output)
        output = self.c7(output)
        output = self.c8(output)

        return output


class AlexNetLight(nn.Module):
    def __init__(self):
        super(AlexNetLight, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 3))),  # 96*26*26
            ('Relu1', nn.ReLU()),
            # ('LRN1', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('Pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # 96*13*13
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ('Conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=2)),  # 256*15*15
            ('Relu2', nn.ReLU()),
            # ('LRN2', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('Pool2', nn.MaxPool2d(kernel_size=(3, 3), stride=2))  # 256*7*7
        ]))
        self.c3 = nn.Sequential(OrderedDict([
            ('Conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1)),  # 384*7*7
            ('Relu3', nn.ReLU()),
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ('Conv4', nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1)),  # 384*7*7
            ('Relu4', nn.ReLU()),
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ('Conv5', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1)),  # 256*7*7
            ('Relu5', nn.ReLU()),
            ('Pool5', nn.MaxPool2d(kernel_size=(3, 3), stride=2))  # 256*3*3
        ]))
        self.c6 = nn.Sequential(OrderedDict([
            ('FullCon6', nn.Linear(in_features=256*3*3, out_features=1024)),
            ('Relu6', nn.ReLU()),
            ('Drop6', nn.Dropout(p=0.5)),
        ]))
        self.c7 = nn.Sequential(OrderedDict([
            ('FullCon7', nn.Linear(in_features=1024, out_features=256)),
            ('Relu7', nn.ReLU()),
            ('Drop7', nn.Dropout(p=0.5)),
        ]))
        self.c8 = nn.Sequential(OrderedDict([
            ('FullCon8', nn.Linear(in_features=256, out_features=10)),
            ('Sig8', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)
        # output = output.view(output.size(0), -1)
        output = torch.flatten(output, 1)
        output = self.c6(output)
        output = self.c7(output)
        output = self.c8(output)

        return output
