import torch
import numpy as np

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="VMI_FGSM", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="VMI_FGSM",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "T": {"desc": "迭代攻击次数", "type": "INT", "required": "true", "def": "50"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "INT", "required": "true", "def": "0.03"},
                                      })

class VMI_FGSM():
    def __init__(self, model, run_device, attack_type='UNTARGETED', tlabel=None, clip_min=0, clip_max=1, T=50, epsilon=16/255):
        self.model = model  # 待攻击的白盒模型
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.device = run_device   #运行设备
        self.T = T  #迭代攻击次数
        self.loss = torch.nn.CrossEntropyLoss()  #定义loss

        self.attack_type = attack_type
        self.target_label = tlabel

    def vmi_fgsm_attack(self, x, y, targeted):
        # 将图像放置GPU中
        x = x.cuda()
        # 将label放置GPU中
        y = np.array(y)
        y = torch.LongTensor(y).to(self.device)
        # 复制原始图像信息
        ori = x.clone()
        # 定义图像可获取梯度 1 224 224 3 RGB tensor
        x.requires_grad = True
        # 定义累计梯度
        sum_grad = 0
        # !VMI 定义vt
        variance = 0
        # 迭代攻击
        for iter in range(self.T):
            #print(iter)
            # 模型预测
            self.model.zero_grad()
            output = self.model(x)
            # 计算loss
            loss = self.loss(output, y)
            if targeted:
                loss = -loss
            # 反向传播
            loss.backward()
            # 得到图像梯度
            grad = x.grad.data
            # !VMI
            new_grad = grad
            # 图像梯度清零
            x.grad = None
            # !VMI-FGSM
            # 定义本轮梯度与var相加
            current_grad = grad + variance
            grad = sum_grad + (current_grad) / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # 更新累计梯度
            sum_grad = grad
            # 计算V（x）。由于输入空间的连续性，无法直接得到前一项，因此可以采样N个样本来近似：
            global_grad = torch.zeros_like(x, requires_grad=False)
            for _ in range(20):
                sample = x.clone().detach()
                sample.requires_grad = True
                rd = (torch.rand_like(x) * 2 - 1) * 1.5 * self.epsilon
                sample = sample + rd.cuda()
                self.model.zero_grad()
                outputs_sample = self.model(sample)
                if targeted:
                    loss_sample = -torch.nn.functional.cross_entropy(outputs_sample, y)
                else:
                    loss_sample = torch.nn.functional.cross_entropy(outputs_sample, y)
                global_grad += torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]

            variance = global_grad / (20 * 1.0) - new_grad

            # 更新图像像素 -1 1 0-255
            x.data = x.data + ((self.epsilon * 2) / self.T) * torch.sign(grad)  # torch.clamp(grad,-2,2)
            # 限制L无穷扰动
            x.data = torch.clamp(x.data - ori.data, -self.epsilon, self.epsilon) + ori.data
            # 限制图像像素范围
            x.data = torch.clamp(x.data, 0, 255)

        return x

    @sefi_component.attack(name="VMI_FGSM", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_labels, target_labels=None):
        if self.attack_type == "UNTARGETED":
            adv_img = self.vmi_fgsm_attack(x=imgs, y=ori_labels, targeted=False)
        else:
            batch_size = imgs.shape[0]
            target_labels = (np.repeat(self.target_label, batch_size)) if target_labels is None else target_labels
            adv_img = self.vmi_fgsm_attack(x=imgs, y=target_labels, targeted=True)
        return adv_img
