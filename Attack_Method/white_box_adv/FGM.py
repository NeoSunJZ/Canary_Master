import torch
import numpy as np

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import cleverhans as ch
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
                                          "norm": {"desc": "范数类型(1, 2, np.inf)", "type": "INT", "def": "2"},
                                          })
class FGM():
    def __init__(self, model, clip_min=0, clip_max=1, epsilon=0.3, norm=2):
        self.model = model  # 待攻击的白盒模型
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm = norm


    @sefi_component.attack(name="FGM", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).contiguous().float()  # 输入img为tensor形式
        img = fast_gradient_method(model_fn=self.model,
                                   x= img,
                                   eps=self.epsilon,
                                   norm=self.norm,
                                   clip_min=self.clip_min,
                                   clip_max=self.clip_max,
                                   y=None,
                                   targeted=False) #非靶向 n_classes为int类型

        return img.cpu().detach().numpy()
