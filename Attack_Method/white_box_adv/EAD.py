import torch
import numpy as np

from foolbox.attacks import EADAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="EAD", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="EAD",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.03"},
                                          "random_start": {"desc": "是否允许在ε球的空间范围初始化", "type": "BOOLEAN", "def": "False"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "steps": {"desc": "更新步骤数", "type": "INT"},
                                          "initial_const": {"desc": "初始权衡常数，用于调整扰动大小的相对重要性和分类的置信度", "type": "FLOAT", "def": "1e-3"},
                                          "binary_search_steps": {"desc": "执行二分搜索时找到扰动范数和分类置信度之间的最佳权衡常数的次数", "type": "INT", "def": "9"},
                                          "initial_stepsize": {"desc": "每个二进制搜索步骤中的优化步骤数", "type": "FLOAT", "def": "1e-2"},
                                          "confidence": {"desc": "将示例标记为对抗性的把握。控制示例和决策边界之间的差距", "type": "FLOAT", "def": "0.0"},
                                          "regularization": {"desc": "控制L1正则化", "type": "FLOAT", "def": "1e-2"},
                                          "decision_rule": {"desc": "选择最佳对抗性示例所依据的规则。它可以最小化L1或ElasticNet距离。", "type": "STRING", "def": "EN"},
                                          "abort_early": {"desc": "设置为TRUE时样本中一旦出现对抗性后将立即停止内部搜索。", "type": "BOOLEAN", "def": "True"},
                                      })
class EAD():
    def __init__(self, model, run_device, epsilon=0.03, attack_type='UNTARGETED', tlabel=1, binary_search_steps=9, steps=10000,
                 initial_stepsize=0.01, confidence=0.0, initial_const=0.001, regularization=0.01, decision_rule='EN',
                 abort_early=True, clip_min=0, clip_max=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = run_device
        self.binary_search_steps = binary_search_steps  # 在对const c进行二进制搜索时要执行的步骤数 整型
        self.steps = steps  # 每个二进制搜索步骤中的优化步骤数 整型
        self.initial_stepsize = initial_stepsize  # 初始步骤大小已更新示例 浮点型
        self.confidence = confidence  # 将示例标记为对抗性所需的置信度 浮点型
        self.initial_const = initial_const  # 开始二进制搜索的const c的初始值 浮点型
        self.regularization = regularization  # 控制L1正则化 浮点型
        self.decision_rule = decision_rule  # 选择最佳对抗示例的规则
        self.abort_early = abort_early  # 一旦找到对抗性示例，请立即停止内部搜索 布尔型
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = fb.PyTorchModel(model, bounds=(self.clip_min, self.clip_max))


    @sefi_component.attack(name="EAD", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        # img = torch.from_numpy(img).to(torch.float32).to(self.device)

        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        # img = ep.astensor(img)
        ori_label = ori_label.squeeze(0)

        # 实例化攻击类
        attack = EADAttack(binary_search_steps=self.binary_search_steps, steps=self.steps, initial_stepsize=self.initial_stepsize, confidence=self.confidence, initial_const=self.initial_const, regularization=self.regularization, decision_rule=self.decision_rule, abort_early=self.abort_early)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attack_type == 'UNTARGETED':
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel]), device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw
        # 由EagerPy张量转化为Native张量

        return adv_img
