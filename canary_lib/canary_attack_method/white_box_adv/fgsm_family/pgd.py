import torch
import numpy as np

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="PGD", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="PGD",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.3"},
                                          "eps_iter": {"desc": "每个攻击迭代的步长", "type": "FLOAT", "def": "0.1"},
                                          "nb_iter": {"desc": "迭代攻击轮数", "type": "INT", "def": "50"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "rand_minmax": {"desc": "支持从连续均匀分布中得到图片上的随机扰动(仅当rand_init为真时才有效)", "type": "FLOAT", "def": "None"},
                                      })
class PGD():
    def __init__(self, model, run_device, epsilon=0.2, eps_iter=0.1, nb_iter=50, clip_min=-3, clip_max=3, rand_init=True,
                 rand_minmax=None, sanity_checks=False, attack_type='UNTARGETED', tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.eps_iter = eps_iter  # 每一轮迭代攻击的步长
        self.nb_iter = nb_iter  # 迭代攻击轮数
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组建的最大浮点值
        self.rand_init = rand_init  # 是否从随机的x开始攻击 布尔型
        self.rand_minmax = rand_minmax  # 支持连续均匀分布，从中得出x上的随机扰动，仅当rand_init为真时才有效 布尔型
        self.sanity_checks = sanity_checks  # 如果为True，则包含断言 布尔型
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    @sefi_component.attack(name="PGD", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, imgs, ori_labels):
        if self.attack_type == 'UNTARGETED':
            adv_img = projected_gradient_descent(model_fn=self.model,
                                             x=imgs,
                                             eps=self.epsilon,
                                             eps_iter=self.eps_iter,
                                             nb_iter=self.nb_iter,
                                             norm=np.inf,
                                             clip_min=self.clip_min,
                                             clip_max=self.clip_max,
                                             y=torch.from_numpy(np.array(ori_labels)).to(self.device),
                                             targeted=False,
                                             rand_init=self.rand_init,
                                             rand_minmax=None,
                                             sanity_checks=self.sanity_checks) #非靶向 n_classes为int类型
            #projected_gradient_descent中 'assert eps_iter <= eps, (eps_iter, eps)'
        else:
            y = torch.from_numpy(np.array(self.tlabel).repeat(imgs.size(0), axis=0)).to(self.device)
            adv_img = projected_gradient_descent(model_fn=self.model,
                                             x=imgs,
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

        return adv_img
