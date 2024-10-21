from canary_lib.canary_defense_method.img_preprocess.DefenseTransformer.model import Conv, UNet
import torch
import torch.nn as nn


class DefenseTransformer:
    def __init__(self, img_size=32,
                 model_ST_path='../canary_lib/canary_defense_method/img_preprocess/DefenseTransformer/resnet56_cifar10.pth',
                 gpu=0, device='cuda'):
        self.model = self.dt_model(img_size, model_ST_path, gpu, device)

    def dt_model(self, img_size, model_ST_path, gpu, device):
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        use_Single_ST = True
        model_ST = UNet(img_size, img_size, block, 3, fwd_out, num_fwd, back_out, num_back, use_Single_ST).to(
            device)
        model_ST.load_state_dict(torch.load(model_ST_path, map_location="cuda:{}".format(gpu)), strict=False)
        return model_ST

    def dt_img(self, img):
        img = self.model(img)
        img = torch.clip(img, 0, 255)
        return img
