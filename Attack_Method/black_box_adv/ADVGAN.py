import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os

import foolbox as fb
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from foolbox.criteria import TargetedMisclassification

sefi_component = SEFIComponent()
models_path = "E:/dataset/adversarial"


@sefi_component.attacker_class(attack_name="ADVGAN")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="ADVGAN",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "model_num_labels": {"desc": "模型中类的数量", "type": "INT", "required": "true", "def": "1000"},
                                          "image_nc": {"desc": "训练数据中图像的channel数", "type": "INT", "required": "true", "def": "3"},
                                          "output_nc": {"desc": "输出的对抗样本中图像的channel数", "type": "INT", "required": "true", "def": "3"},
                                          "box_min": {"desc": "", "type": "FLOAT", "required": "true"},
                                          "box_max": {"desc": "", "type": "FLOAT", "required": "true"},
                                          "lr": {"desc": "optimizer的学习率", "type": "FLOAT", "required": "true", "def": "0.001"},
                                          "epochs": {"desc": "学习轮数", "type": "INT", "required": "true", "def": "60"},
                                      })
class ADVGAN():
    def __init__(self, model, run_device, model_num_labels=1000, image_nc=3, output_nc=3, box_min=0, box_max=1, lr=0.001, epochs=60):
        self.model = model  # 待攻击的模型
        self.device = run_device  # 一般是cuda
        self.model_num_labels = model_num_labels  # 模型中类的数量
        self.input_nc = image_nc  # 训练数据中图像的channel数
        self.output_nc = output_nc  # 输出的对抗样本中图像的channel数
        self.box_min = box_min
        self.box_max = box_max
        self.lr = lr  # 学习率
        self.epochs = epochs  # 学习轮数

        self.gen_input_nc = image_nc
        self.netG = Generator(self.gen_input_nc, image_nc).to(self.device)
        self.netDisc = Discriminator(image_nc).to(self.device)

        # 初始化权重
        self.netG.apply(self.weights_init)
        self.netDisc.apply(self.weights_init)

        # 初始化 optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=self.lr)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    # 权重初始化函数
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # 正态分布
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)  # 全为零

    # 从
    @staticmethod
    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return adv_images


    @staticmethod
    def train(self, img, ori_labels, epochs):
        for epoch in range(1, epochs+1):

            images, labels = img, ori_labels
            images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.train_batch(self, images, labels)

        return adv_images

    @sefi_component.attack(name="ADVGAN", is_inclass=True, support_model=[], attack_type="BLACK_BOX")
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        ori_label = ori_label.squeeze(0).numpy()
        ori_label = torch.as_tensor(ori_label).cuda()

        advgan = self.train(self, img, ori_label, self.epochs)

        return advgan


# 生成器
class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,  # 进入时的channel数
                 image_nc,  # 出来时的channel数
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        ]

        bottle_neck_lis = [ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32), ]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# 判别器
class Discriminator(nn.Module):
    def __init__(self, image_nc):  # channel数
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


# resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
