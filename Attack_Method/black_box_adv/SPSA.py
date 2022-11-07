import torch
import numpy as np

from cleverhans.torch.attacks.spsa import spsa

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="SPSA")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="SPSA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.3"},
                                          "norm": {"desc": "范数类型(1, 2, np.inf)", "type": "INT", "def": "np.inf"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "early_stop_loss_threshold": {"desc": "损失低于此阈值时停止（默认为None）", "type": "FLOAT"},
                                          "nb_iter": {"desc": "优化迭代次数", "type": "INT"},
                                          "learning_rate": {"desc": "优化器学习率", "type": "FLOAT"},
                                          "delta": {"desc": "用于SPSA近似的扰动大小", "type": "FLOAT"},
                                          "spsa_samples": {"desc": "一次评估的图片数量", "type": "INT"},
                                          "spsa_iters": {"desc": "更新前模型评估的次数（每次评估都会在不同的spsa_samples数量的输入上评估）", "type": "INT"},
                                      })
class SPSA():
    def __init__(self, model, run_device, clip_min=-3, clip_max=3, epsilon=0.3, norm=np.inf, attack_type='UNTARGETED',
                 tlabel=-1, nb_iter=100,early_stop_loss_threshold=None, learning_rate=0.01, delta=0.01,
                 spsa_samples=128, spsa_iters=1):
        self.model = model  # 待攻击的白盒模型
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm = norm
        self.nb_iter = nb_iter
        self.early_stop_loss_threshold = early_stop_loss_threshold
        self.learning_rate = learning_rate
        self.delta = delta
        self.spsa_samples = spsa_samples # Number of inputs to evaluate at a single time. The true batch size(the number of evaluated inputs for each update) is `spsa_samples * spsa_iters
        self.spsa_iters = spsa_iters # Number of model evaluations before performing an update, where each evaluation is on spsa_samples different inputs

    @sefi_component.attack(name="SPSA", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_labels):
        if self.attack_type == 'UNTARGETED':
            adv_img = spsa(model_fn=self.model,
                       x=imgs,
                       eps=self.epsilon,
                       nb_iter=self.nb_iter,
                       norm=self.norm,
                       clip_min=self.clip_min,
                       clip_max=self.clip_max,
                       y=torch.from_numpy(np.array(ori_labels)).to(self.device),
                       targeted=False,
                       early_stop_loss_threshold=self.early_stop_loss_threshold,
                       learning_rate=self.learning_rate,
                       delta=self.delta,
                       spsa_samples=self.spsa_samples,
                       spsa_iters=self.spsa_iters,
                       is_debug=False,
                       sanity_checks=False) #非靶向 n_classes为int类型
        elif self.attack_type == 'TARGETED':
            y = torch.from_numpy(np.array(self.tlabel).repeat(imgs.size(0), axis=0)).to(self.device)
            adv_img = spsa(model_fn=self.model,
                       x=imgs,
                       eps=self.epsilon,
                       nb_iter=self.nb_iter,
                       norm=self.norm,
                       clip_min=self.clip_min,
                       clip_max=self.clip_max,
                       y=y,
                       targeted=True,
                       early_stop_loss_threshold=self.early_stop_loss_threshold,
                       learning_rate=self.learning_rate,
                       delta=self.delta,
                       spsa_samples=self.spsa_samples,
                       spsa_iters=self.spsa_iters,
                       is_debug=False,
                       sanity_checks=False)  # 非靶向 n_classes为int类型
        else:
            raise Exception("未知攻击方式")

        return adv_img
