import torch
import random as rd
import numpy as np
import scipy.stats as st
from torch.nn import functional as F
from torch import nn
from torchvision.transforms import Resize

from canary_lib.canary_attack_method.white_box_adv.deep_fusing.deep_fusing_model import deepFusingModel
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="DEEP_FUSING", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(
    handler_target=ComponentType.ATTACK, name="DEEP_FUSING",
    handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "T": {"desc": "攻击轮数", "type": "INT", "required": "true", "def": "50"},
        "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "required": "true", "def": "0.06274"},
    })
class deepFusing():
    def __init__(self, run_device, attack_type='UNTARGETED', T=50, epsilon=16, model=None, tlabel=None):
        self.device = run_device
        # 定义攻击迭代轮数
        self.T = T
        # 定义无穷扰动范围
        self.epsilon = epsilon + 4
        if self.epsilon <= 4:
            self.epsilon = 4
        # 定义损失函数
        self.loss_CEP = torch.nn.CrossEntropyLoss()  # 交叉熵
        self.criterion_KL = self.KLLoss()  # KL散度
        # 定义存储原始图片与靶向图片特征图
        self.ori_label_output = []
        self.ori_feature_output = []
        # 定义存储原始图片的label
        self.ori_label = []
        # 定义模型
        self.models = ["resnet50", "inception_v3", "vgg16", "densenet161", "mobilenet_v2"]

    # 对非残差网络侵蚀
    def getDropout_keeppro(self, num):
        if num == 1:  # inceptionV3
            Dropout_keeppro = float(int(rd.uniform(0.002, 0.06) * 1000)) / 1000  # Dropout层的保留参数
        elif num == 2:  # VGG16
            Dropout_keeppro = float(int(rd.uniform(0.002, 0.6) * 1000)) / 1000  # Dropout层的保留参数
        elif num == 4:  # MobilenetV2
            Dropout_keeppro = float(int(rd.uniform(0.002, 0.09) * 1000)) / 1000  # Dropout层的保留参数
        return Dropout_keeppro

    # 对残差网络侵蚀
    def getSkip_keeppro(self, num):
        if num == 0:  # resnet50
            skip_pro = float(int(rd.uniform(1 - float(int(rd.uniform(0.00001, 0.07) * 1000)) / 1000, 1 + float(
                int(rd.uniform(0.0001, 0.05) * 1000)) / 1000) * 1000)) / 1000  # skip跳跃连接的扰动参数
        elif num == 3:  # densenet161
            skip_pro = float(int(rd.uniform(1 - float(int(rd.uniform(0.00001, 0.1) * 1000)) / 1000, 1 + float(
                int(rd.uniform(0.0001, 0.1) * 1000)) / 1000) * 1000)) / 1000  # skip跳跃连接的扰动参数
        return skip_pro

    # 获取侵蚀参数
    def getEroding_parameter(self, num):
        if num == 0:  # resnet50
            return self.getSkip_keeppro(num)
        elif num == 1:  # inceptionV3
            return self.getDropout_keeppro(num)
        elif num == 2:  # Vgg16
            return self.getDropout_keeppro(num)
        elif num == 3:  # Densenet161
            return self.getSkip_keeppro(num)
        elif num == 4:  # MobilenetV2
            return self.getDropout_keeppro(num)

    # 保证模型不被侵蚀
    def getNoEroding_parameter(self, num):
        if num == 0:  # resnet50
            return 1.0
        elif num == 1:  # inceptionV3
            return 0.0
        elif num == 2:  # Vgg16
            return 0.0
        elif num == 3:  # Densenet161
            return 1.0
        elif num == 4:  # MobilenetV2
            return 0.0

    # input——diversity
    def input_diversity(self, input_tensor):
        shape = input_tensor.shape
        rnd = torch.Tensor(1, 1).uniform_(shape[-2] + 30, shape[-1] + 30).to(
            torch.device(self.device)).int()  # 返回一个均匀分布的矩阵

        rescaled = torch.nn.functional.interpolate(input_tensor,
                                                   size=[rnd.data.cpu().numpy()[0][0], rnd.data.cpu().numpy()[0][0]],
                                                   mode='nearest')

        h_rem = shape[-2] + 30 - rnd.data.cpu().numpy()[0][0]
        w_rem = shape[-1] + 30 - rnd.data.cpu().numpy()[0][0]
        pad_top = torch.Tensor(1, 1).uniform_(0, h_rem).to(self.device).int()
        pad_bottom = h_rem - pad_top.cpu().numpy()[0][0]
        pad_left = torch.Tensor(1, 1).uniform_(0, w_rem).to(self.device).int()
        pad_right = w_rem - pad_left.cpu().numpy()[0][0]
        padded = torch.nn.functional.pad(rescaled, (
        pad_left.cpu().numpy()[0][0], pad_right, pad_top.cpu().numpy()[0][0], pad_bottom), "constant", value=0)
        padded = torch.nn.functional.interpolate(padded,
                                                 size=[shape[-2], shape[-1]],
                                                 mode='bilinear')

        if torch.Tensor(1, 1).uniform_(0, 1).data.cpu().numpy()[0][0] < 0.7:
            ret = padded
        else:
            ret = input_tensor

        return ret

    class KLLoss(nn.Module):
        def forward(self, map_pred,
                    map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
            self.epsilon = 1e-8
            map_pred = map_pred.float()
            map_gtd = map_gtd.float()

            map_pred = map_pred.view(1, -1)  # change the map_pred into a tensor with n rows and 1 cols
            map_gtd = map_gtd.view(1, -1)  # change the map_pred into a tensor with n rows and 1 cols

            min1 = torch.min(map_pred)
            max1 = torch.max(map_pred)
            map_pred = (map_pred - min1) / (
                        max1 - min1 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

            min2 = torch.min(map_gtd)
            max2 = torch.max(map_gtd)
            map_gtd = (map_gtd - min2) / (
                        max2 - min2 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

            map_pred = map_pred / (torch.sum(
                map_pred) + self.epsilon)  # normalization step to make sure that the map_pred sum to 1
            map_gtd = map_gtd / (
                        torch.sum(map_gtd) + self.epsilon)  # normalization step to make sure that the map_gtd sum to 1

            KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
            KL = map_gtd * KL
            KL = torch.sum(KL)

            return KL

    class attack_method_3(object):
        class GaussianBlurConv(torch.nn.Module):
            def __init__(self, kernellen=3, sigma=4, channels=3):
                super().__init__()
                if kernellen % 2 == 0:
                    kernellen += 1
                self.kernellen = kernellen
                kernel = self.gkern(kernellen, sigma).astype(np.float32)
                self.channels = channels
                kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
                kernel = np.repeat(kernel, self.channels, axis=0)
                self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

            def forward(self, x):
                padv = (self.kernellen - 1) // 2
                x = nn.functional.pad(x, pad=(padv, padv, padv, padv), mode='replicate')
                x = F.conv2d(x, self.weight, stride=1, padding=0, groups=self.channels)
                return x

            def gkern(self, kernlen=3, nsig=4):
                """Returns a 2D Gaussian kernel array."""
                interval = (2 * nsig + 1.) / (kernlen)
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
                kern1d = np.diff(st.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                return kernel

        def saliency_attack_start(self, image, image0, epsilon, data_grad, pre_mom, T, miu):  #
            miu_factor = miu  # 0.5
            unit_vector_data_grad = torch.abs(data_grad.contiguous().view(1, -1))
            grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
            unit_vector_data_grad = data_grad / grad_mean
            next_grad = miu_factor * pre_mom + (1 - miu_factor) * data_grad
            noise = next_grad

            noise = noise / (torch.std(torch.abs(noise), dim=1, keepdim=True) + 1e-12)
            # noise = noise / (torch.std(torch.abs(noise), dim=1, keepdim=True) + 1e-12)

            # 高斯卷积
            gaussianBlurConv = self.GaussianBlurConv(kernellen=7).cuda()
            noise = torch.transpose(noise, 1, 3).contiguous()
            noise = gaussianBlurConv(noise)
            noise = torch.transpose(noise, 1, 3).contiguous()

            image = image.data + (epsilon * 1.5) / T * torch.clamp(torch.round(noise), -2, 2)
            image.data = torch.clamp((image - image0), -epsilon, epsilon) + image0.data
            image.data = torch.clamp(image, 0, 255)

            return image, next_grad

        def saliency_attack(self, image, image0, epsilon, data_grad, pre_mom, sum_mom, T, miu):
            miu_factor = miu  # 0.5

            unit_vector_data_grad = torch.abs(data_grad.contiguous().view(1, -1))

            grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
            unit_vector_data_grad = data_grad / grad_mean
            next_grad = miu_factor * pre_mom + (1 - miu_factor) * unit_vector_data_grad
            First_mom = next_grad

            sec_mom = torch.sqrt(sum_mom + 1e-8)
            sec_mom = sec_mom * 1e5
            next_grad = next_grad / (sec_mom + 1e-2)

            #trick
            noise = next_grad
            noise = noise / (torch.std(torch.abs(noise), dim=1, keepdim=True) + 1e-12)
            noise = noise / (torch.std(torch.abs(noise), dim=1, keepdim=True) + 1e-12)

            # 高斯卷积
            gaussianBlurConv = self.GaussianBlurConv(kernellen=7).cuda()
            noise = torch.transpose(noise, 1, 3).contiguous()
            noise = gaussianBlurConv(noise)
            noise = torch.transpose(noise, 1, 3).contiguous()

            image = image.data + (epsilon * 1.5) / T * torch.clamp(torch.round(noise), -2, 2)
            image.data = torch.clamp((image - image0), -epsilon, epsilon) + image0.data
            image.data = torch.clamp(image, 0, 255)
            return image, First_mom

    @sefi_component.attack(name="DEEP_FUSING", is_inclass=True, support_model=[], attack_type="WHITE_BOX", model_require=False)
    def attack(self, imgs, ori_labels, tlabels=None):
        advs = []
        for img in imgs:
            # 拷贝原始图片
            img = torch.from_numpy(img).to(self.device).float()
            img = img.permute(2, 0, 1)
            resize = Resize([500, 500])
            img = resize(img)
            img = img.permute(1, 2, 0)
            img = img.unsqueeze(axis=0)

            img = img.clone()
            ori_img = img.clone()

            # 定义图片可求导
            img.requires_grad = True
            # 定义loss对象
            loss = 0
            # 初始化累计梯度----
            for batch, model_name in enumerate(self.models):
                model = deepFusingModel(device=self.device, name=model_name,
                                        dropout_keeppro=self.getNoEroding_parameter(batch))
                print(img.shape)
                output, feature = model(img)
                # 获取标签
                self.ori_label_output.append(
                    torch.from_numpy(torch.argmax(output).data.cpu().numpy()).unsqueeze(0).to(self.device).long())
                # 获取特征图
                self.ori_feature_output.append(feature)
                # 计算loss
                loss1 = self.criterion_KL(feature, self.ori_feature_output[-1].detach())
                loss2 = self.loss_CEP(output, self.ori_label_output[-1]) * 10
                loss += loss1 + loss2
                model.model.zero_grad()
            print('ori label:', "[Resnet50:", self.ori_label_output[0].item(), " , InceptionV3:",
                  self.ori_label_output[1].item(), " , VGG16:", self.ori_label_output[2].item(), " , Densenet161:",
                  self.ori_label_output[3].item(), " , MoilenetV2:", self.ori_label_output[4].item(), "]")
            # 反向传播
            loss.backward()
            # 获取图像梯度
            data_grad = img.grad.data
            # 定义初始化梯度
            temp_prev_grad_init = 0
            # 调用显著性攻击方法
            attack_perform = self.attack_method_3()
            # 返回值为扰动数据、累计梯度
            perturbed_data, temp_prev_grad = attack_perform.saliency_attack_start(img, ori_img, self.epsilon, data_grad,
                                                                                  temp_prev_grad_init, self.T, 0)

            # 定义累计梯度计量
            sum_mom_init = torch.pow(temp_prev_grad, 1)
            img.data = perturbed_data.data
            temp_prev_grad_step = temp_prev_grad
            sum_mom_step = torch.pow(sum_mom_init, 2)
            sum_mom_nodecay = torch.pow(sum_mom_init, 2)
            # 迭代攻击----(T-1)
            for iter in range(self.T - 1):
                loss = 0
                pre_label = []
                # 遍历模型
                for batch, model_name in enumerate(self.models):
                    model = deepFusingModel(device=self.device, name=model_name,
                                            dropout_keeppro=self.getEroding_parameter(batch))
                    # DIM
                    adv_jzt = torch.transpose(img, 1, 3).contiguous()
                    adv_jzt = self.input_diversity(adv_jzt)
                    adv_jzt = torch.transpose(adv_jzt, 1, 3).contiguous()
                    # 模型进行预测,预测原图片
                    output, featrue_adv = model(adv_jzt)
                    # 记录当前label
                    pre_label.append(torch.argmax(output).item())
                    # 计算loss
                    loss1 = self.criterion_KL(self.ori_feature_output[batch].detach(), featrue_adv)
                    loss2 = self.loss_CEP(output, self.ori_label_output[batch]) * 10
                    loss += loss1 + loss2
                    # 模型梯度清零
                    model.model.zero_grad()
                    del model
                # 打印
                print('iter:', iter + 1, "[Resnet50:", pre_label[0], " , InceptionV3:", pre_label[1], " , VGG16:",
                      pre_label[2], " , Densenet161:", pre_label[3], " , MoilenetV2:", pre_label[4], "]")
                # 反向传播
                loss.backward()
                # 获取梯度
                data_step_grad = img.grad.data
                sum_mom_step = torch.pow(sum_mom_step, 1) + torch.pow(data_step_grad, 2)
                # 计算显著性攻击
                perturbed_data_step, temp_prev_grad_step = attack_perform.saliency_attack(img, ori_img, self.epsilon,
                                                                                          data_step_grad,
                                                                                          temp_prev_grad_step, sum_mom_step,
                                                                                          self.T, 0)
                img.data = perturbed_data_step.data
                adv = img.data.cpu().numpy()[0]
                advs.append(adv)
        return advs