import os
import random
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import eagerpy as ep
from torch.utils.data import DataLoader

from Attack_Method.black_box_adv.adv_gan import models
from Attack_Method.black_box_adv.adv_gan.adv_gan_core import AdvGAN_Attack
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="AdvGan")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="AdvGan",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "model_num_labels": {"desc": "模型中类的数量", "type": "INT", "required": "true", "def": "1000"},
                                          "image_nc": {"desc": "训练数据中图像的channel数", "type": "INT", "required": "true", "def": "3"},
                                          "output_nc": {"desc": "输出的对抗样本中图像的channel数", "type": "INT", "required": "true", "def": "3"},
                                          "box_min": {"desc": "", "type": "FLOAT", "required": "true"},
                                          "box_max": {"desc": "", "type": "FLOAT", "required": "true"},
                                          "lr": {"desc": "optimizer的学习率", "type": "FLOAT", "required": "true", "def": "0.001"},
                                          "epochs": {"desc": "学习轮数", "type": "INT", "required": "true", "def": "60"},
                                      })
class AdvGan():
    def __init__(self, model, run_device, attack_type="UNTARGETED", model_num_labels=1000, image_nc=3, clip_min=0, clip_max=1, epochs=60):
        self.model = model  # 待攻击的模型
        self.device = run_device  # 一般是cuda
        self.model_num_labels = model_num_labels  # 模型中类的数量
        self.gen_input_nc = image_nc
        self.input_nc = image_nc  # 训练数据中图像的channel数
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.epochs = epochs  # 学习轮数

        self.models_path = os.path.dirname(__file__) + "/weight/"
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        self.pretrained_G = None

    @sefi_component.attack_init(name="AdvGan")
    def universal_perturbation(self, dataset, batch_size, model_name):
        self.netG_file_name = self.models_path + "netG_epoch_{}_{}.pth".format(self.epochs, model_name)
        if os.path.exists(self.netG_file_name):
            return
        advGAN = AdvGAN_Attack(device=self.device,
                               model=self.model,
                               model_num_labels=self.model_num_labels,
                               image_nc=self.input_nc,
                               box_min=self.clip_min,
                               box_max=self.clip_max)

        advGAN.train(dataset, batch_size, self.epochs, netG_file_name=self.netG_file_name)

    @sefi_component.attack(name="AdvGan", is_inclass=True, support_model=[])
    def attack(self, imgs, ori_labels, tlabels=None):
        if self.pretrained_G is None:
            self.pretrained_G = models.Generator(self.gen_input_nc, self.input_nc).to(self.device)
            self.pretrained_G.load_state_dict(torch.load(self.netG_file_name))
            self.pretrained_G.eval()

        perturbations = self.pretrained_G(imgs)
        # perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_imgs = perturbations + imgs
        return adv_imgs

