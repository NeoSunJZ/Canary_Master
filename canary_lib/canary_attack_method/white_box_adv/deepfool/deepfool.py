import foolbox
import torch
import numpy as np
import eagerpy as ep

from foolbox import TargetedMisclassification
from canary_lib.canary_attack_method.white_box_adv.deepfool.deepfool_core import L2DeepFoolAttack, LinfDeepFoolAttack
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="DeepFool", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="DeepFool",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "pixel_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "pixel_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT",
                                                          "selector": [{"value": "TARGETED", "name": "靶向"},
                                                                       {"value": "UNTARGETED", "name": "非靶向"}],
                                                          "required": "true"},
                                          "p": {"desc": "范数类型", "type": "SELECT", "selector": [{"value": "l-2", "name": "l-2"}, {"value": "l-inf", "name": "l-inf"}], "required": "true"},
                                          "max_iter": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "1000"},
                                          "num_classes": {"desc": "模型中类的数量", "type": "INT", "required": "true"},
                                          "overshoot": {"desc": "最大超出边界的值", "type": "FLOAT", "def": "0.02"},
                                          "candidates": {"desc": "最大超出边界的值", "type": "INT", "def": "10"},
                                      })
class DeepFool():
    def __init__(self, model, run_device, attack_type="UNTARGETED", pixel_min=0, pixel_max=1, p="l-2", overshoot=0.02, max_iter=50, candidates=10, num_classes=1000):
        self.model = foolbox.PyTorchModel(model, bounds=(pixel_min, pixel_max), device=run_device)
        self.attack_type = attack_type
        self.p = p

        self.device = run_device

        self.overshoot = overshoot  # 边界超出量，用作终止条件以防止类别更新
        self.max_iter = max_iter # FoolBox的最大迭代次数
        self.candidates = candidates
        self.loss = 'logits'

        # 全局变量
        self.classes = None
        self.p_total = None
        self.loop_count = 0

    @sefi_component.attack(name="DeepFool", is_inclass=True, support_model=[])
    def attack(self, imgs, ori_labels, tlabels=None):

        ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        imgs = ep.astensor(imgs)
        # 实例化攻击类
        if self.p == "l-2":
            attack = L2DeepFoolAttack(steps=self.max_iter, candidates=self.candidates, overshoot=self.overshoot, loss=self.loss)
        else:
            attack = LinfDeepFoolAttack(steps=self.max_iter, candidates=self.candidates, overshoot=self.overshoot, loss=self.loss)

        if self.attack_type == 'UNTARGETED':
            raw, clipped, is_adv = attack(self.model, imgs, ori_labels, epsilons=None)
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor(tlabels))  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, imgs, ori_labels, epsilons=None, criterion=criterion)

        adv_img = raw.raw
        self.p_total = attack.p_total
        self.loop_count = attack.loop_count

        return adv_img
