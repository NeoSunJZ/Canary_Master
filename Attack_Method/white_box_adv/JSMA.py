import torch
import numpy as np
from torch.autograd import Variable

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="JSMA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="JSMA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "epsilon": {"desc": "以无穷范数作为约束，设置最大值", "def": "0.03", "required": "true"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"}, {"value": "UNTARGETED", "name": "非靶向"}], "def": "TARGETED"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "epochs": {"desc": "迭代攻击轮数", "type": "INT", "def": "1000"},
                                          "theta": {"desc": "扰动系数", "type": "FLOAT", "def": "0.3"},
                                          "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "def": "-3.0"},
                                          "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "def": "3.0"}})
class JSMA():
    def __init__(self, model,epsilon=0.03, attacktype='TARGETED', tlabel=1, epochs=1000, theta=0.3, clip_min=-3.0, clip_max=3.0):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs  # 迭代攻击轮数
        self.theta = theta  # 扰动系数
        self.clip_min = clip_min  # 对抗样本像素下界(与模型相关)
        self.clip_max = clip_max  # 对抗样本像素上界(与模型相关)


    @sefi_component.attack(name="JSMA", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):

        img = Variable(torch.from_numpy(img).to(self.device).float())

        #使用预测模式 主要影响dropout和BN层的行为
        a = self.model(img).data.cpu().numpy()

        def saliency_map(F, x, t, mask):
            # pixel influence on target class
            # 对函数进行反向传播，计算输出变量关于输入变量的梯度
            F[0,t].backward(retain_graph=True)
            # F[0,t]的梯度为正表明是对t的正向贡献、为负表明是对t的负向贡献
            derivative = x.grad.data.cpu().numpy().copy()

            # 超限的点遮罩mask为0，不在参与修改
            alphas = derivative * mask

            # 返回一个用-1填充的跟输入 形状和类型 一致的数组
            # 简化了bstas的计算
            betas = -np.ones_like(alphas)

            sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)
            # find optimal pixel & direction of perturbation
            # 寻找最有利的攻击点（对目标t负向贡献的那些点）
            idx = np.argmin(sal_map)

            #转换成(p1,p2)格式
            idx = np.unravel_index(idx, mask.shape) # 还原为像素位置坐标

            # 符号，指明变化方向
            pix_sign = np.sign(alphas)[idx]

            return idx, pix_sign

        # 图像数据梯度可以获取
        img.requires_grad = True

        # 设置为不保存梯度值 自然也无法修改
        for param in self.model.parameters():
            param.requires_grad = False

        target = Variable(torch.Tensor([float(self.tlabel)]).to(self.device).long())

        loss_func = torch.nn.CrossEntropyLoss()

        # the mask defines the search domain
        # each modified pixel with border value is set to zero in mask
        mask = np.ones_like(img.data.cpu().numpy())

        for epoch in range(self.epochs):

            # forward
            output = self.model(img)

            label = np.argmax(output.data.cpu().numpy())
            loss = loss_func(output, target)

            # 如果定向攻击成功
            if label == self.tlabel:
                break

            # 梯度清零
            if img.grad is not None:
                img.grad.zero_()

            idx, pix_sign = saliency_map(output, img, self.tlabel, mask)

            # apply perturbation
            img.data[idx] = img.data[idx] + pix_sign * self.theta * (self.clip_max - self.clip_min)

            # 达到极限的点不再参与更新
            if (img.data[idx] <= self.clip_min) or (img.data[idx] >= self.clip_max):
                mask[idx] = 0
                b = img.data[idx].cpu()
                img.data[idx] = np.clip(b, self.clip_min, self.clip_max)

        return img
