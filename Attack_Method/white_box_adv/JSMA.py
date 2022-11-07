import torch
import numpy as np
from torch.autograd import Variable

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="JSMA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="JSMA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "alpha": {"desc": "攻击算法的学习率(每轮攻击步长)", "type": "FLOAT", "def": "5e-3"},
                                          "epsilon": {"desc": "扰动大小", "type": "FLOAT"},
                                          "pixel_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "pixel_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "T": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "1000"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class JSMA():
    def __init__(self, model, run_device, T=1000, pixel_min=-3.0, pixel_max=3.0, epsilon=0.3, tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.T = int(T)  # 迭代攻击轮数
        self.pixel_min = float(pixel_min)  # 像素值的下限
        self.pixel_max = float(pixel_max)  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.epsilon = float(epsilon)  # 扰动系数
        self.tlabel = int(tlabel)
        self.device = run_device

    # 实现saliency_map
    # F:model x:img t:target mask:搜索空间
    def saliency_map(self, F, x, t, mask):
        # pixel influence on target class
        # 对函数进行反向传播，计算输出变量关于输入变量的梯度
        F[0, t].backward(retain_graph=True)
        # F[0,t]的梯度为正表明是对t的正向贡献、为负表明是对t的负向贡献
        derivative = x.grad.data.cpu().numpy().copy()

        # 超限的点遮罩mask为0，不再参与修改
        alphas = derivative * mask

        # 返回一个用-1填充的跟输入 形状和类型 一致的数组
        # 简化了bstas的计算
        betas = -np.ones_like(alphas)

        sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)
        # find optimal pixel & direction of perturbation
        # 寻找最有利的攻击点（对目标t负向贡献的那些点）
        idx = np.argmin(sal_map)

        # 转换成(p1,p2)格式
        idx = np.unravel_index(idx, mask.shape)  # 还原为像素位置坐标

        # 符号，指明变化方向
        pix_sign = np.sign(alphas)[idx]

        return idx, pix_sign

    @sefi_component.attack(name="JSMA", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        # 图像数据梯度可以获取
        img.requires_grad = True

        # 设置为不保存梯度值 自然也无法修改
        for param in self.model.parameters():
            param.requires_grad = False

        # 攻击目标
        target = Variable(torch.Tensor([float(self.tlabel)]).to(self.device).long())

        mask = np.ones_like(img.data.cpu().numpy())

        for epoch in range(self.T):

            # forward
            output = self.model(img)

            label = np.argmax(output.data.cpu().numpy())
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(output, target)
            print("epoch={} label={} loss={}".format(epoch, label, loss))

            # 如果定向攻击成功
            if label == self.tlabel:
                break

            # 梯度清零
            if img.grad is not None:
                img.grad.zero_()

            idx, pix_sign = self.saliency_map(output, img, self.tlabel, mask)

            # apply perturbation
            img.data[idx] = img.data[idx] + pix_sign * self.epsilon * (self.pixel_max - self.pixel_min)

            # 达到极限的点不再参与更新
            if (img.data[idx] <= self.pixel_min) or (img.data[idx] >= self.pixel_max):
                print("idx={} over {}".format(idx, img.data[idx]))
                mask[idx] = 0
                b = img.data[idx].cpu()
                img.data[idx] = np.clip(b, self.pixel_min, self.pixel_max)

        return img