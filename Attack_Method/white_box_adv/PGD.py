import torch
import numpy as np

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="PGD", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="PGD",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "eps": {"desc": "以无穷范数作为约束，设置最大值", "required": "true"},
                                          "eps_iter": {"desc": "每一轮迭代攻击的步长", "required": "true"},
                                          "nb_iter": {"desc": "迭代攻击轮数", "required": "true"},
                                          "norm": {"desc": "范数顺序", "type": "SELECT", "selector": [{"value": "np.inf"}, {"value": "1"}, {"value": "2"}], "def": "np.inf"},
                                          "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "def": "None", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "def": "None", "required": "true"},
                                          "random_init": {"desc": "是否从随机的x开始攻击", "type": "BOOL", "def": "True"},
                                          "random_minmax": {"desc": "支持连续均匀分布，从中得出x上的随机扰动，仅当rand_init为真时才有效", "type": "BOOL", "def": "None"},
                                          "sanity_checks": {"desc": "若为真包含断言", "type": "BOOL", "def": "True"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"}, {"value": "UNTARGETED", "name": "非靶向"}], "def": "TARGETED"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class PGD():
    def __init__(self, model, epsilon=0.2, eps_iter=0.1, nb_iter=50, norm=np.inf, clip_min=-3, clip_max=3, rand_init=True,
                 rand_minmax=None, sanity_checks=True, attacktype='UNTARGETED', tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.eps_iter = eps_iter  # 每一轮迭代攻击的步长
        self.nb_iter = nb_iter  # 迭代攻击轮数
        self.norm = norm  # 范数顺序
        self.clip_min = clip_min  # 对抗样本像素下界(与模型相关)
        self.clip_max = clip_max  # 对抗样本像素上界(与模型相关)
        self.rand_init = rand_init  # 是否从随机的x开始攻击 布尔型
        self.rand_minmax = rand_minmax  # 支持连续均匀分布，从中得出x上的随机扰动，仅当rand_init为真时才有效 布尔型
        self.sanity_checks = sanity_checks  # 若为真则包含断言 布尔型
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @sefi_component.attack(name="PGD", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()  # 输入img为tensor形式
        y = torch.tensor([self.tlabel], device=self.device)  # 具有真实标签的张量 默认为None

        if self.attacktype == 'UNTARGETED':
            img = projected_gradient_descent(model_fn=self.model,
                                             x=img,
                                             eps=self.epsilon,
                                             eps_iter=self.eps_iter,
                                             nb_iter=self.nb_iter,
                                             norm=self.norm,
                                             clip_min=self.clip_min,
                                             clip_max=self.clip_max,
                                             y=None,
                                             targeted=False,
                                             rand_init=self.rand_init,
                                             rand_minmax=None,
                                             sanity_checks=self.sanity_checks) #非靶向 n_classes为int类型
            #projected_gradient_descent中 'assert eps_iter <= eps, (eps_iter, eps)'
        else:
            img = projected_gradient_descent(model_fn=self.model,
                                             x=img,
                                             eps=self.epsilon,
                                             eps_iter=self.eps_iter,
                                             nb_iter=self.nb_iter,
                                             norm=self.norm,
                                             clip_min=self.clip_min,
                                             clip_max=self.clip_max,
                                             y=y,
                                             targeted=True,
                                             rand_init=self.rand_init,
                                             rand_minmax=None,
                                             sanity_checks=self.sanity_checks) #靶向 y带有真标签的张量

        return img
