import torch
import numpy as np

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="FGM", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="FGM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "df_v": "None"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "df_v": "None"},
                                          "sanity_checks": {"desc": "若为真则包含断言；关闭以使用较少的运行时/内存，或用于故意传递奇怪输入的单元测试", "type": "BOOL", "df_v": "False"},
                                          "attack_type": {"desc": "攻击类型(靶向(TARGETED) / 非靶向(UNTARGETED))", "type": "STR", "df_v": "'UNTARGETED'"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class FGM():
    def __init__(self, model, clip_min=-3, clip_max=3, sanity_checks=False, epsilon=0.2, attacktype='UNTARGETED', tlabel=-1):
        self.model = model  # 待攻击的白盒模型
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.sanity_checks = sanity_checks  # 如果为True，包含断言；关闭以使用较少的运行时/内存，或用于故意传递奇怪输入的单元测试
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @sefi_component.attack(name="FGM", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()  # 输入img为tensor形式
        y = torch.tensor([self.tlabel], device=self.device)  # 具有真实标签的张量 默认为None

        if self.attacktype == 'UNTARGETED':
            img = fast_gradient_method(model_fn=self.model,
                                       x=img,
                                       eps=self.epsilon,
                                       norm=np.inf,
                                       clip_min=self.clip_min,
                                       clip_max=self.clip_max,
                                       y=None,
                                       targeted=False,
                                       sanity_checks=self.sanity_checks) #非靶向 n_classes为int类型
        else:
            img = fast_gradient_method(model_fn=self.model,
                                       x=img,
                                       eps=self.epsilon,
                                       norm=np.inf,
                                       clip_min=self.clip_min,
                                       clip_max=self.clip_max,
                                       y=y,
                                       targeted=True,
                                       sanity_checks=self.sanity_checks) #靶向 y带有真标签的张量

        return img
