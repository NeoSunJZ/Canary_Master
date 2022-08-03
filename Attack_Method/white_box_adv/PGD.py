import torch
import numpy as np

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="PGD", perturbation_budget_var_name="epsilon")
class PGD():
    def __init__(self, model, epsilon=0.2, eps_iter=0.1, nb_iter=50, clip_min=-3, clip_max=3, rand_init=True,
                 rand_minmax=None, sanity_checks=True, attacktype='untargeted', tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.eps_iter = eps_iter  # 每一轮迭代攻击的步长
        self.nb_iter = nb_iter  # 迭代攻击轮数
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组建的最大浮点值
        self.rand_init = rand_init  # 是否从随机的x开始攻击 布尔型
        self.rand_minmax = rand_minmax  # 支持连续均匀分布，从中得出x上的随机扰动，仅当rand_init为真时才有效 布尔型
        self.sanity_checks = sanity_checks  # 如果为True，则包含断言 布尔型
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @sefi_component.attack(name="PGD", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()  # 输入img为tensor形式
        y = torch.tensor([self.tlabel], device=self.device)  # 具有真实标签的张量 默认为None

        if self.attacktype == 'untargeted':
            img = projected_gradient_descent(model_fn=self.model,
                                             x=img,
                                             eps=self.epsilon,
                                             eps_iter=self.eps_iter,
                                             nb_iter=self.nb_iter,
                                             norm=np.inf,
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
                                             norm=np.inf,
                                             clip_min=self.clip_min,
                                             clip_max=self.clip_max,
                                             y=y,
                                             targeted=True,
                                             rand_init=self.rand_init,
                                             rand_minmax=None,
                                             sanity_checks=self.sanity_checks) #靶向 y带有真标签的张量

        return img
