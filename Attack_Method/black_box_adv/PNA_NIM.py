import torch
import numpy as np
from functools import partial

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="PNA_NIM", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="PNA_NIM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "T": {"desc": "迭代攻击次数", "type": "INT", "required": "true", "def": "50"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "INT", "required": "true", "def": "0.03"},
                                      })
class PNA_NIM():
    def __init__(self, model, run_device, clip_min=0, clip_max=1, T=50, epsilon=0.03):
        self.model = model
        self.device = run_device
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.T = T
        self.epsilon = epsilon
        self.drop_hook_func = partial(self.attn_drop_mask_grad, gamma=0)

    def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
        mask = torch.ones_like(grad_in[0]) * gamma
        return (mask * grad_in[0][:],)


    @sefi_component.attack(name="PNA_NIM", is_inclass=True, support_model=[], attack_type="BLACK_BOX")
    def attack(self, img, ori_label):
        loss_ = torch.nn.CrossEntropyLoss()

        ori_label = np.array(ori_label)
        ori_label = torch.LongTensor(ori_label).to(self.device)

        # 复制原始图像信息
        ori = img.clone()
        # 定义图像可获取梯度 1 224 224 3 RGB tensor
        img.requires_grad = True
        # !NI
        adv_ni_grad = 0
        # 定义累计梯度
        sum_grad = 0
        # 迭代攻击
        for iter in range(self.T):
            # 记录总输出，求和形式
            output = 0
            # 模型预测
            self.model.zero_grad()
            output = self.model(img + adv_ni_grad * ((self.epsilon * 2) / self.T))
            # 计算loss
            loss = loss_(output, ori_label)

            # 反向传播
            loss.backward()
            # 得到图像梯度
            grad = img.grad.data
            # 图像梯度清零
            img.grad = None
            # !MIM
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # 与之前的sum_grad相加
            grad = sum_grad + grad
            # 相加后均一化
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # 更新累计梯度
            sum_grad = grad
            # !NIM
            adv_ni_grad = sum_grad

            # 更新图像像素 -1 1 0-1
            img.data = img.data + ((self.epsilon * 2) / self.T) * torch.sign(grad)  # torch.clamp(grad,-2,2)
            # 限制L无穷扰动
            img.data = torch.clamp(img.data - ori.data, -self.epsilon, self.epsilon) + ori.data
            # 限制图像像素范围
            img.data = torch.clamp(img.data, self.clip_min, self.clip_max)

        return img