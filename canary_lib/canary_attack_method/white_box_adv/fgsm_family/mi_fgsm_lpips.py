import numpy as np
import torch
import lpips

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="MI_FGSM_LPIPS", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(
    handler_target=ComponentType.ATTACK, name="MI_FGSM_LPIPS",
    handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true", "def": "0.00"},
        "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true", "def": "1.00"},
        "epoch": {"desc": "攻击轮数", "type": "INT", "required": "true", "def": "50"},
        "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "required": "true", "def": "0.06274"},
        "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
        "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}
    })
class MI_FGSM_LPIPS():
    def __init__(self, model, run_device, attack_type='UNTARGETED', clip_min=0, clip_max=1, epoch=50, epsilon= 4 / 255, tlabel=None):
        self.model = model  # 待攻击的白盒模型
        self.epoch = int(epoch)
        self.device = run_device
        self.epsilon = float(epsilon)  # 以无穷范数作为约束，设置最大
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.attack_type = attack_type
        self.tlabel = tlabel

    def clip_value(self, x, ori_x):
        x = torch.clamp((x - ori_x), -self.epsilon, self.epsilon) + ori_x
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x.data

    @sefi_component.attack(name="MI_FGSM_LPIPS", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, ori_labels, tlabels=None):
        batch_size = img.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        ori = img.clone()
        img.requires_grad = True

        # 定义累计梯度
        sum_grad = torch.zeros_like(img)

        loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        loss_ = torch.nn.CrossEntropyLoss()

        for i in range(int(self.epoch)):
            loss_lpips = loss_fn_vgg(img, ori) * 100
            loss_lpips = loss_lpips.sum() / img.shape[0]

            self.model.zero_grad()

            output = self.model(img)
            if self.attack_type == 'UNTARGETED':
                loss_ces = loss_(output, torch.Tensor(ori_labels).to(self.device).long())  # 非靶向
            else:
                loss_ces = -loss_(output, torch.Tensor(tlabels).to(self.device).long())  # 靶向

            all_loss = loss_ces - loss_lpips
            all_loss.backward()
            grad = img.grad.data
            img.grad = None

            # 执行mim算法
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            grad = grad + sum_grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = grad

            img.data = img.data + ((self.epsilon * 2) / self.epoch) * torch.sign(grad)
            img.data = self.clip_value(img, ori)
        return img