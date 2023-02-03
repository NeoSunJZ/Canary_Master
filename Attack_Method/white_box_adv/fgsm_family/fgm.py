import numpy as np
import torch

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="FGM")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="FGM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.3"},
                                          })
class FGM():
    def __init__(self, model, run_device, attack_type='TARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=0.3):
        self.model = model  # 待攻击的白盒模型
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值

        self.attack_type = attack_type
        self.target_label = tlabel
        self.device = run_device

    @sefi_component.attack(name="FGM", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_labels, target_labels=None):
        if self.attack_type == 'UNTARGETED':
            adv_img = fast_gradient_method(model_fn=self.model,
                                       x=imgs,
                                       eps=self.epsilon,
                                       norm=2,
                                       clip_min=self.clip_min,
                                       clip_max=self.clip_max,
                                       y=torch.from_numpy(np.array(ori_labels)).to(self.device),
                                       targeted=False)
        elif self.attack_type == 'TARGETED':
            batch_size = imgs.shape[0]
            target_labels = (np.repeat(self.target_label, batch_size)) if target_labels is None else target_labels
            adv_img = fast_gradient_method(model_fn=self.model,
                                       x=imgs,
                                       eps=self.epsilon,
                                       norm=2,
                                       clip_min=self.clip_min,
                                       clip_max=self.clip_max,
                                       y=torch.from_numpy(np.array(target_labels)).to(self.device),
                                       targeted=True)
        else:
            raise RuntimeError("[ Logic Error ] Illegal target type!")
        return adv_img
