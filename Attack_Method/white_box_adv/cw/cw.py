import time

import torch

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="CW", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="CW",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "classes": {"desc": "模型中类的数量", "type": "INT", "required": "true"},

                                          "clip_min": {"desc": "像素值的下限", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "像素值的上限", "type": "FLOAT", "required": "true"},

                                          "initial_const": {"desc": "初始权衡常数，权衡扰动范数和分类置信度在损失中的权重", "type": "FLOAT",
                                                            "def": "1e-3"},

                                          "binary_search_steps": {"desc": "二分查找最大次数(寻找扰动范数和分类置信度之间的最佳权衡常数c)",
                                                                  "type": "INT", "def": "5"},
                                          "lr": {"desc": "学习率", "type": "FLOAT", "def": "5e-3"},
                                          "max_iterations": {"desc": "最大迭代次数", "type": "INT", "def": "1000"},

                                          "attack_type": {"desc": "攻击类型", "type": "SELECT",
                                                          "selector": [{"value": "TARGETED", "name": "靶向"},
                                                                       {"value": "UNTARGETED", "name": "非靶向"}],
                                                          "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class CW:
    def __init__(self, model, run_device, classes=1000, lr=5e-3, clip_min=0, clip_max=1, initial_const=1e-2,
                 binary_search_steps=5, max_iterations=1000, attack_type='UNTARGETED', tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.n_classes = classes  # 模型中类的数量

        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）

        self.initial_const = initial_const  # 初始权衡常数，权衡扰动范数和分类置信度在损失中的权重
        self.lr = lr  # 攻击算法的学习速率（浮点数）
        self.binary_search_steps = binary_search_steps  # 二分查找最大次数（寻找扰动范数和分类置信度之间的最佳权衡常数c）

        self.max_iterations = max_iterations  # 最大迭代次数 整型
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel  # 靶向攻击目标标签
        self.device = run_device

    @sefi_component.attack(name="CW", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_label):
        print(imgs.shape)
        if self.attack_type == 'UNTARGETED':
            adv_imgs = carlini_wagner_l2(model_fn=self.model,
                                    x=imgs,
                                    n_classes=self.n_classes,
                                    y=None,
                                    targeted=False,
                                    lr=self.lr,
                                    clip_min=self.clip_min,
                                    clip_max=self.clip_max,
                                    initial_const=self.initial_const,
                                    binary_search_steps=self.binary_search_steps,
                                    max_iterations=self.max_iterations)
        else:
            adv_imgs = carlini_wagner_l2(model_fn=self.model,
                                    x=imgs,
                                    n_classes=self.n_classes,
                                    y=torch.tensor([self.tlabel], device=self.device),
                                    targeted=True,
                                    lr=self.lr,
                                    clip_min=self.clip_min,
                                    clip_max=self.clip_max,
                                    initial_const=self.initial_const,
                                    binary_search_steps=self.binary_search_steps,
                                    max_iterations=self.max_iterations)
        return adv_imgs
