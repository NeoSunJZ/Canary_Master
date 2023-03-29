import torch
import numpy as np
import eagerpy as ep
from foolbox import PyTorchModel
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.criteria import TargetedMisclassification, Misclassification
from Attack_Method.black_box_adv.hop_skip_jump_attack.hop_skip_jump_attack_core import HopSkipJumpAttack as FoolBoxHopSkipJumpAttack
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="HopSkipJumpAttack", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="HopSkipJumpAttack",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "ARRAY_INT"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "gamma": {"desc": "设置二分搜索停止条件的阈值（如果二分两点的距离小于此值，则认为已经搜索到边界）", "type": "FLOAT", "def": "0.1"},
                                          "steps": {"desc": "循环次数", "type": "INT", "def": "64"},
                                          "max_gradient_eval_steps": {"desc": "估计梯度时最大的估计次数", "type": "INT", "def": "10000"},
                                          "initial_gradient_eval_steps": {"desc": "用于估计梯度方向的向量数量", "type": "INT", "def": "100"},
                                          "constraint": {"desc": "范数", "type": "SELECT", "selector": [{"value": "l2", "name": "l2范数"},{"value": "linf", "name": "l∞范数"}], "required": "true"}
                                      })
class HopSkipJumpAttack:
    def __init__(self, model, run_device, attack_type='UNTARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=None,
                 steps=64,
                 initial_gradient_eval_steps=100,
                 max_gradient_eval_steps=10000,
                 gamma=0.1,
                 constraint="linf"):
        self.model = PyTorchModel(model, bounds=(clip_min, clip_max), device=run_device)
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel

        self.init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)  # 攻击用来寻找起点
        self.steps = steps  # 二叉搜索最大步数（让xt到bound）
        self.initial_gradient_eval_steps = initial_gradient_eval_steps  # 方法用蒙特卡洛模拟梯度（多个方向的点用于计算梯度）
        self.max_gradient_eval_steps = max_gradient_eval_steps  # Maximum number of evaluations for gradient estimation.
        self.gamma = gamma  # 设置二分搜索停止条件的阈值：如果二分两点的距离小于此值，则认为已经搜索到边界
        self.constraint = constraint  # 范数

    @sefi_component.attack(name="HopSkipJumpAttack", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, imgs, ori_labels, tlabels=None):
        batch_size = imgs.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        # 转为PyTorch变量
        tlabels = ep.astensor(torch.from_numpy(np.array(tlabels)).to(self.device))
        ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        imgs = ep.astensor(imgs)

        # 实例化攻击类
        attack = FoolBoxHopSkipJumpAttack(init_attack=self.init_attack,
                                          steps=self.steps,
                                          initial_gradient_eval_steps=self.initial_gradient_eval_steps,
                                          max_gradient_eval_steps=self.max_gradient_eval_steps,
                                          gamma=self.gamma,
                                          tensorboard=False,
                                          constraint=self.constraint)
        if self.attack_type == 'UNTARGETED':
            criterion = Misclassification(labels=ori_labels)
            raw, clipped, is_adv = attack(self.model, imgs, criterion, epsilons=None)
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=tlabels)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, imgs, criterion, epsilons=None)

        # 由EagerPy张量转化为Native张量
        adv_img = raw.raw
        return adv_img
