import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from .nets.densenet import densenet121, densenet161, densenet169, densenet201
from .nets.inception import inception_v3
from .nets.vgg import vgg11, vgg13, vgg16, vgg19
from .nets.mobilenetv2 import mobilenet_v2
from .nets.resnet import resnet101, resnet152, resnet18, resnet34, resnet50, wide_resnet101_2, wide_resnet50_2

networks = {
    "resnet50": resnet50,
    "inception_v3": inception_v3,
    "vgg16": vgg16,
    "densenet161" : densenet161,
    "mobilenet_v2": mobilenet_v2
}

class deepFusingModel(nn.Module):
    def __init__(self, device, name, dropout_keeppro=0.5):
        super(deepFusingModel, self).__init__()
        self.device = device
        self.mean=torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device).float()
        self.std=torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device).float()
        self.name = name

        torchvision_model_class = networks[name]
        self.model = torchvision_model_class(pretrained=True, keep_pro=dropout_keeppro).to(self.device).eval()

    def init_img(self, img):  # 将图片转化为标准化的图片
        img = img / 255.0
        img = (img - self.mean) / self.std
        img = torch.transpose(img, 2, 3).contiguous()
        img = torch.transpose(img, 1, 2).contiguous()
        img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
        return img

    def forward(self,img):
        return self.model(self.init_img(img))