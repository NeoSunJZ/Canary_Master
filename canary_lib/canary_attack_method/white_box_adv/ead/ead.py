import numpy as np
import torch

from canary_lib.canary_attack_method.white_box_adv.ead.ead_core import EAD

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="EAD", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="EAD",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "num_classes": {"desc": "模型中类的数量", "type": "INT", "required": "true"},
                                          "clip_min": {"desc": "像素值的下限", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "像素值的上限", "type": "FLOAT", "required": "true"},
                                          "init_const": {"desc": "初始权衡常数，权衡扰动范数和分类置信度在损失中的权重", "type": "FLOAT", "def": "1e-3"},
                                          "binary_search_steps": {"desc": "二分查找最大次数(寻找扰动范数和分类置信度之间的最佳权衡常数c)", "type": "INT", "def": "5"},
                                          "lr": {"desc": "学习率", "type": "FLOAT", "def": "5e-3"},
                                          "max_iters": {"desc": "最大迭代次数", "type": "INT", "def": "1000"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT",
                                                          "selector": [{"value": "TARGETED", "name": "靶向"},
                                                                       {"value": "UNTARGETED", "name": "非靶向"}],
                                                          "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "kappa": {"desc": "将示例标记为对抗性的把握，控制示例和决策边界之间的差距", "type": "FLOAT", "required": "true", "def": "0.0"},
                                          "beta": {"desc": "控制L1正则化", "type": "INT", "required": "true", "def": "1e-3"},
                                          "EN": {"desc": "选择最佳对抗性示例所依据的规则，它可以最小化L1或ElasticNet距离", "type": "BOOL", "required": "true", "def": "True"}
                                      })
class EADAttack:
    def __init__(self, model, run_device, attack_type='TARGETED', tlabel=None, clip_min=0, clip_max=1, kappa=0, init_const=0.001, lr=0.02, binary_search_steps=5, max_iters=10000, beta=0.001, EN=True, num_classes=1000):
        self.model = model  # 待攻击的白盒模型
        self.device = run_device
        self.attack_type = attack_type
        self.lower_bound = clip_min  # 像素值的下限
        self.upper_bound = clip_max  # 像素值的上限
        self.tlabel = tlabel
        self.kappa = kappa
        self.lr = lr
        self.init_const = init_const
        self.max_iters = max_iters
        self.binary_search_steps = binary_search_steps
        self.beta = beta
        self.EN = EN
        self.num_classes = num_classes

        if self.attack_type == 'TARGETED':
            self.targeted = True
        elif self.attack_type == 'UNTARGETED':
            self.targeted = False
        else:
            raise RuntimeError("[ Logic Error ] Illegal target type!")

    @sefi_component.attack(name="EAD", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_label, tlabels=None):
        ead = EAD(model=self.model, targeted=self.targeted, kappa=self.kappa, init_const=self.init_const, lr=self.lr, binary_search_steps=self.binary_search_steps, max_iters=self.max_iters, lower_bound=self.lower_bound, upper_bound=self.upper_bound, beta=self.beta, EN=self.EN, num_classes=self.num_classes)
        imgs = imgs.cpu().numpy()
        length = len(imgs)

        if self.attack_type == 'TARGETED':
            tlabel = np.repeat(self.tlabel, length) if tlabels is None else tlabels
        elif self.attack_type == 'UNTARGETED':
            tlabel = ori_label
        else:
            raise RuntimeError("[ Logic Error ] Illegal target type!")

        adv_imgs = ead.batch_perturbation(imgs, tlabel, length, self.device)
        adv_imgs = torch.tensor(adv_imgs, dtype=torch.float).to(self.device)
        return adv_imgs
