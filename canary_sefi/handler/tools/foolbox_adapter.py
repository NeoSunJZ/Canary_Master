import torch
import numpy as np
from foolbox import PyTorchModel
from foolbox.criteria import TargetedMisclassification, Misclassification
import eagerpy as ep
from canary_sefi.core.component.component_decorator import SEFIComponent
sefi_component = SEFIComponent()


class FoolboxAdapter:
    def __init__(self, model, foolbox_attack, attack_target, required_epsilon):
        self.model = model
        self.foolbox_attack = foolbox_attack
        self.attack_target = attack_target
        self.required_epsilon = required_epsilon
        self.device, self.attack_type, self.target_label, self.epsilon = None, None, None, None

    def init_args(self, run_device, attack_type, target_label, clip_min, clip_max, epsilon):
        self.model = PyTorchModel(self.model, bounds=(clip_min, clip_max), device=run_device)
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.target_label = target_label
        self.epsilon = epsilon

    def attack(self, imgs, ori_labels, target_labels=None):
        batch_size = imgs.shape[0]

        ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        imgs = ep.astensor(imgs)

        if self.required_epsilon is True and self.epsilon is None:
            epsilon = 16/255
        else:
            epsilon = self.epsilon

        raw, clipped = None, None
        if self.attack_type == 'UNTARGETED':
            if 'UNTARGETED' not in self.attack_target:
                raise RuntimeError("[ Logic Error ] Illegal target type!")
            criterion = Misclassification(labels=ori_labels)
            raw, clipped, is_adv = self.foolbox_attack(self.model, imgs, epsilons=epsilon, criterion=criterion)
        elif self.attack_type == 'TARGETED':
            if 'TARGETED' not in self.attack_target:
                raise RuntimeError("[ Logic Error ] Illegal target type!")
            target_labels = (np.repeat(self.target_label, batch_size)) if target_labels is None else target_labels
            target_labels = ep.astensor(torch.from_numpy(np.array(target_labels)).to(self.device))
            criterion = TargetedMisclassification(target_classes=target_labels)
            raw, clipped, is_adv = self.foolbox_attack(self.model, imgs, epsilons=epsilon, criterion=criterion)
        else:
            raise RuntimeError("[ Logic Error ] Illegal target type!")

        # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的有效性
        if self.epsilon is None:
            return raw.raw  # 由EagerPy张量转化为Native张量
        else:
            return clipped.raw  # 由EagerPy张量转化为Native张量
