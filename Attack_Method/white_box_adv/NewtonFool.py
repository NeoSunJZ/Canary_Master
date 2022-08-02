import torch
import numpy as np

from foolbox.attacks import NewtonFoolAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="NewtonFool", perturbation_budget_var_name="epsilon")
class NewtonFool():
    def __init__(self, model, epsilon=0.03, attacktype='untargeted', tlabel=1, steps=50, stepsize=6/25):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.steps = steps  # 要执行的更新步骤数 整型
        self.stepsize = stepsize  # 每个更新步骤的大小 浮点型

    @sefi_component.attack(name="NewtonFool", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        # 模型预处理
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        bounds = (-3, 3)
        fmodel = fb.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing)

        # 攻击前数据处理
        ori_label = np.array([ori_label])
        img = torch.from_numpy(img).to(torch.float32)

        ori_label = ep.astensor(torch.LongTensor(ori_label))
        img = ep.astensor(img)

        # 准确性判断
        fb.utils.accuracy(fmodel, inputs=img, labels=ori_label)

        # 实例化攻击类
        attack = NewtonFoolAttack(steps=self.steps, stepsize=self.stepsize)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attacktype == 'untargeted':
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel]),
                                                  device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
