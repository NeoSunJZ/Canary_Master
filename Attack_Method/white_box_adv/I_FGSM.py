import torch
import numpy as np

from foolbox.attacks import L2BasicIterativeAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="I_FGSM", perturbation_budget_var_name="epsilon")
class I_FGSM():
    def __init__(self, model, epsilon=0.03, attacktype='untargeted', tlabel=1, rel_stepsize=6/25, abs_stepsize=None,
                 steps=50, random_start=False):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rel_stepsize = rel_stepsize  # 相对于epsilon的步长 浮点型
        self.abs_stepsize = abs_stepsize  # 如果给定，优先于rel_stepsize 浮点型
        self.steps = steps  # 更新步骤数 整型
        self.random_start = random_start  # 控制是否在允许的epsilon ball中随机启动 布尔型


    @sefi_component.attack(name="I_FGSM", is_inclass=True, support_model=["vision_transformer"])
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
        attack = L2BasicIterativeAttack( rel_stepsize=self.rel_stepsize, abs_stepsize=self.abs_stepsize, steps=self.steps, random_start=self.random_start)
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
