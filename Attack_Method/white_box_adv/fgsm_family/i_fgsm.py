import torch
import numpy as np

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="I_FGSM", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="I_FGSM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.3"},
                                          "eps_iter": {"desc": "每个攻击迭代的步长", "type": "FLOAT", "def": "0.1"},
                                          "nb_iter": {"desc": "迭代攻击轮数", "type": "INT", "def": "50"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "norm": {"desc": "范数", "type": "FLOAT", "def": "np.inf"},
                                      })
class I_FGSM():
    def __init__(self, model, run_device, epsilon=0.2, eps_iter=0.1, nb_iter=50, clip_min=-3, clip_max=3, rand_init=False,
                 sanity_checks=False, attack_type='UNTARGETED', tlabel=1, norm=np.inf):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.eps_iter = eps_iter  # 每一轮迭代攻击的步长
        self.nb_iter = nb_iter  # 迭代攻击轮数
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组建的最大浮点值
        self.rand_init = rand_init  # 是否从随机的x开始攻击 布尔型
        self.sanity_checks = sanity_checks  # 如果为True，则包含断言 布尔型

        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.target_labels = tlabel
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm = norm

    @sefi_component.attack(name="I_FGSM", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, imgs, ori_labels, target_labels=None):
        # 与PGD用的同一个函数，当rand_init为False时，为I_FGSM方法
        if self.attack_type == 'UNTARGETED':
            adv_img = projected_gradient_descent(model_fn=self.model,
                                             x=imgs,
                                             eps=self.epsilon,
                                             eps_iter=self.eps_iter,
                                             nb_iter=self.nb_iter,
                                             norm=self.norm,
                                             clip_min=self.clip_min,
                                             clip_max=self.clip_max,
                                             y=torch.from_numpy(np.array(ori_labels)).to(self.device),
                                             targeted=False,
                                             rand_init=self.rand_init,
                                             rand_minmax=None,
                                             sanity_checks=self.sanity_checks) #非靶向 n_classes为int类型
        elif self.attack_type == 'TARGETED':
            batch_size = imgs.shape[0]
            target_labels = (np.repeat(self.target_label, batch_size)) if target_labels is None else target_labels
            adv_img = projected_gradient_descent(model_fn=self.model,
                                             x=imgs,
                                             eps=self.epsilon,
                                             eps_iter=self.eps_iter,
                                             nb_iter=self.nb_iter,
                                             norm=self.norm,
                                             clip_min=self.clip_min,
                                             clip_max=self.clip_max,
                                             y=torch.from_numpy(np.array(target_labels)).to(self.device),
                                             targeted=True,
                                             rand_init=self.rand_init,
                                             rand_minmax=None,
                                             sanity_checks=self.sanity_checks) #靶向 y带有真标签的张量

        else:
            raise RuntimeError("[ Logic Error ] Illegal target type!")
        return adv_img
