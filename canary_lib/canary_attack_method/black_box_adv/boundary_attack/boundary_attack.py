import torch
import numpy as np
import eagerpy as ep
from foolbox import PyTorchModel
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.criteria import TargetedMisclassification, Misclassification
from canary_lib.canary_attack_method.black_box_adv.boundary_attack.boundary_attack_core import BoundaryAttack as FoolBoxBoundaryAttack
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="BA", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="BA",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "ARRAY_INT"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "max_iterations": {"desc": "要运行的最大步骤数，可能会在此之前收敛并停止", "type": "INT", "def": "25000"},
                                          "spherical_step": {"desc": "正交（球形）补偿的初始步长", "type": "FLOAT", "def": "1e-2"},
                                          "source_step": {"desc": "迈向目标的步骤的初始步长", "type": "FLOAT", "def": "1e-2"},
                                          "source_step_convergence": {"desc": "设置停止条件的阈值：如果在攻击期间source_step小于此值，则攻击已收敛并将停止", "type": "FLOAT", "def": "1e-07"},
                                          "step_adaptation": {"desc": "步长乘以或除以的因子", "type": "FLOAT", "def": "1.5"},
                                          "update_stats_every_k": {"desc": "每k步检查是否更新spherical_step，source_step", "type": "INT", "def": "10"},
                                      })
class BoundaryAttack:
    def __init__(self, model, run_device, attack_type='UNTARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=None,
                 max_iterations=25000,
                 spherical_step=0.01,
                 source_step=0.01,
                 source_step_convergence=1e-07,
                 step_adaptation=1.5,
                 update_stats_every_k=10):
        self.model = PyTorchModel(model, bounds=(clip_min, clip_max), device=run_device)
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)  # 攻击用来寻找起点
        self.steps = max_iterations  # 要运行的最大步骤数，可能会在此之前收敛并停止 整型
        self.spherical_step = spherical_step  # 正交（球形）补偿的初始步长
        self.source_step = source_step  # 迈向目标的步骤的初始步长 浮点型
        self.source_step_convergence = source_step_convergence  # 设置停止条件的阈值：如果在攻击期间source_step小于此值，则攻击已收敛并将停止
        self.step_adaptation = step_adaptation  # 步长乘以或除以的因子 浮点型
        self.update_stats_every_k = update_stats_every_k  # 整型

    @sefi_component.attack(name="BA", is_inclass=True, support_model=[])
    def attack(self, imgs, ori_labels, tlabels=None):
        batch_size = imgs.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        # 转为PyTorch变量
        tlabels = ep.astensor(torch.from_numpy(np.array(tlabels)).to(self.device))
        ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        imgs = ep.astensor(imgs)

        # 实例化攻击类
        attack = FoolBoxBoundaryAttack(init_attack=self.init_attack, steps=self.steps,
                                       spherical_step=self.spherical_step,
                                       source_step=self.source_step,
                                       source_step_convergence=self.source_step_convergence,
                                       step_adaptation=self.step_adaptation,
                                       tensorboard=False,
                                       update_stats_every_k=self.update_stats_every_k)
        if self.attack_type == 'UNTARGETED':
            criterion = Misclassification(labels=ori_labels)
            raw, clipped, is_adv = attack(self.model, imgs, criterion, epsilons=None)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=tlabels)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, imgs, criterion, epsilons=None)

        # 由EagerPy张量转化为PyTorch Native张量
        adv_img = raw.raw
        return adv_img
