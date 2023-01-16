import torch
import numpy as np

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="VMI_FGSM", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="VMI_FGSM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "T": {"desc": "迭代攻击次数", "type": "INT", "required": "true", "def": "50"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "INT", "required": "true", "def": "0.03"},
                                      })

class VMI_FGSM():
    def __init__(self, model, run_device, clip_min=0, clip_max=1, T=50, epsilon=0.03):
        self.model = model  # 待攻击的白盒模型
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.device = run_device   #运行设备
        self.T = T  #迭代攻击次数
        self.loss = torch.nn.CrossEntropyLoss()  #定义loss

    @sefi_component.attack(name="VMI_FGSM", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, label):
        # 将图像放置GPU中
        img = img.cuda()
        # 将label放置GPU中
        label = np.array(label)
        label = torch.LongTensor(label).to(self.device)
        # 复制原始图像信息
        ori = img.clone()
        # 定义图像可获取梯度 1 224 224 3 RGB tensor
        img.requires_grad = True
        # 定义累计梯度
        sum_grad = 0
        # !VMI 定义vt
        variance = 0
        # 迭代攻击
        for iter in range(self.T):
            print(iter)
            # 记录总输出，求和形式
            output = 0
            # 模型预测
            self.model.zero_grad()
            output = self.model(img)
            # 计算loss
            loss = self.loss(output, label)
            # 反向传播
            loss.backward()
            # 得到图像梯度
            grad = img.grad.data
            # !VMI
            new_grad = grad
            # 图像梯度清零
            img.grad = None
            # !VMI-FGSM
            # 定义本轮梯度与var相加
            current_grad = grad + variance
            grad = sum_grad + (current_grad) / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # 更新累计梯度
            sum_grad = grad
            # 计算V（x）。由于输入空间的连续性，无法直接得到前一项，因此可以采样N个样本来近似：
            global_grad = torch.zeros_like(img, requires_grad=False)
            for _ in range(20):
                sample = img.clone().detach()
                sample.requires_grad = True
                rd = (torch.rand_like(img) * 2 - 1) * 1.5 * self.epsilon
                sample = sample + rd.cuda()
                self.model.zero_grad()
                outputs_sample = self.model(sample)
                loss_sample = torch.nn.functional.cross_entropy(outputs_sample, label)
                global_grad += torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]

            variance = global_grad / (20 * 1.0) - new_grad

            # 更新图像像素 -1 1 0-255
            img.data = img.data + ((self.epsilon * 2) / self.T) * torch.sign(grad)  # torch.clamp(grad,-2,2)
            # 限制L无穷扰动
            img.data = torch.clamp(img.data - ori.data, -self.epsilon, self.epsilon) + ori.data
            # 限制图像像素范围
            img.data = torch.clamp(img.data, 0, 255)

        return img