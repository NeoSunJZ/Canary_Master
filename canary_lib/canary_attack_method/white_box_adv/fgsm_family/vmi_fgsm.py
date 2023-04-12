import torch
import numpy as np

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="VMI_FGSM", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(
    handler_target=ComponentType.ATTACK, name="VMI_FGSM",
    handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true", "def": "0.00"},
        "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true", "def": "1.00"},
        "T": {"desc": "迭代攻击次数", "type": "INT", "required": "true", "def": "10"},
        "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "required": "true", "def": "0.06274"},
        "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
        "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}
    })
class VMI_FGSM():
    def __init__(self, model, run_device, attack_type='UNTARGETED', clip_min=0, clip_max=1, T=10, epsilon=16/255, tlabel=None):
        self.model = model  # 待攻击的白盒模型
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = run_device

    # 将图片进行clip
    def clip_value(self, x, ori_x):
        if self.epsilon is not None:
            x = torch.clamp((x - ori_x), -self.epsilon, self.epsilon) + ori_x
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x.data

    @sefi_component.attack(name="VMI_FGSM", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, ori_labels, tlabels=None):
        batch_size = img.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        loss_ = torch.nn.CrossEntropyLoss()

        # 定义图片可获取梯度
        img.requires_grad = True
        # 克隆原始数据
        ori_img = img.clone()
        sum_grad = 0  # 累计梯度

        # VMI Variance
        variance = 0
        # 迭代攻击
        for iter in range(self.T):
            self.model.zero_grad()
            output = self.model(img)

            # 计算loss
            if self.attack_type == 'UNTARGETED':
                loss = loss_(output, torch.Tensor(ori_labels).to(self.device).long())  # 非靶向
            else:
                loss = -loss_(output, torch.Tensor(tlabels).to(self.device).long())  # 靶向
            # 反向传播
            loss.backward()
            grad = img.grad.data
            img.grad = None
            # VMI
            new_grad = grad
            # 定义本轮梯度与Variance相加
            current_grad = grad + variance
            grad = sum_grad + (current_grad) / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # 更新累计梯度
            sum_grad = grad
            # 计算V（x）。由于输入空间的连续性，无法直接得到前一项，因此可以采样N个样本来近似
            global_grad = torch.zeros_like(img, requires_grad=False)
            for _ in range(20):
                sample = img.clone().detach()
                sample.requires_grad = True
                rd = (torch.rand_like(img) * 2 - 1) * 1.5 * self.epsilon
                sample = sample + rd.to(self.device)
                self.model.zero_grad()
                outputs_sample = self.model(sample)
                if self.attack_type == 'UNTARGETED':
                    loss_sample = loss_(outputs_sample, torch.Tensor(ori_labels).to(self.device).long())  # 非靶向
                else:
                    loss_sample = -loss_(outputs_sample, torch.Tensor(tlabels).to(self.device).long())  # 靶向
                global_grad += torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
            variance = global_grad / (20 * 1.0) - new_grad

            # 更新图像像素 -1 1 0-1
            img.data = img.data + ((self.epsilon * 2) / self.T) * torch.sign(grad)
            img.data = self.clip_value(img, ori_img)
        return img
