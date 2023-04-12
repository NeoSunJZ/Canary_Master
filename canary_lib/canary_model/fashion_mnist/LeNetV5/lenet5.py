import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)),
            ('Relu1', nn.ReLU()),
            ('Pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ('Conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))),
            ('Relu2', nn.ReLU()),
            ('Pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # Output:16*5*5
        ]))
        self.c3 = nn.Sequential(OrderedDict([
            ('FullCon3', nn.Linear(in_features=400, out_features=120)),
            ('Relu3', nn.ReLU()),
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ('FullCon4', nn.Linear(in_features=120, out_features=84)),
            ('Relu4', nn.ReLU()),
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ('FullCon5', nn.Linear(in_features=84, out_features=10)),
            ('Sig5', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)

        output = output.view(-1, 16*5*5)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)

        return output


