import torch
import numpy as np

from foolbox.attacks import EADAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="EAD", perturbation_budget_var_name="epsilon")
class EAD():
    def __init__(self, model, epsilon=0.03, attacktype='untargeted', tlabel=1, binary_search_steps=9, steps=10000,
                 initial_stepsize=0.01, confidence=0.0, initial_const=0.001, regularization=0.01,decision_rule='EN',
                 abort_early=True):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.binary_search_steps = binary_search_steps  # 在对const c进行二进制搜索时要执行的步骤数 整型
        self.steps = steps  # 每个二进制搜索步骤中的优化步骤数 整型
        self.initial_stepsize = initial_stepsize  # 初始步骤大小已更新示例 浮点型
        self.confidence = confidence  # 将示例标记为对抗性所需的置信度 浮点型
        self.initial_const = initial_const  # 开始二进制搜索的const c的初始值 浮点型
        self.regularization = regularization  # 控制L1正则化 浮点型
        self.decision_rule = decision_rule  # 选择最佳对抗示例的规则
        self.abort_early = abort_early  # 一旦找到对抗性示例，请立即停止内部搜索 布尔型


    @sefi_component.attack(name="EAD", is_inclass=True, support_model=["vision_transformer"])
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
        attack = EADAttack(binary_search_steps=self.binary_search_steps, steps=self.steps, initial_stepsize=self.initial_stepsize, confidence=self.confidence, initial_const=self.initial_const, regularization=self.regularization, decision_rule=self.decision_rule, abort_early=self.abort_early)
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
