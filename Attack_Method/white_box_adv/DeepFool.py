import torch
import numpy as np

from foolbox.attacks import L2DeepFoolAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="DeepFool", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="DeepFool",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.03"},
                                          "max_iterations": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "1000"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "candidates": {"desc": "限制应考虑的最可能类的数量", "type": "INT"},
                                          "overshoot": {"desc": "最大超出边界的值", "type": "FLOAT", "def": "0.02"},
                                          "loss": {"desc": "每个更新步骤的大小", "type": "FLOAT", "def": "logits"},
                                      })
class DeepFool():
    def __init__(self, model, epsilon=0.03, attack_type='UNTARGETED', tlabel=1, max_iterations=50, candidates=10,
                 overshoot=0.02, loss='logits', clip_min=0, clip_max=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.steps = max_iterations  # 要执行的最大步骤数
        self.candidates = candidates  # 限制应考虑的最可能类的数量
        self.overshoot = overshoot  # 超出边界的量 浮点型
        self.loss = loss  # (Union[typing_extensions.Literal['logits'], typing_extensions.Literal['crossentropy']]) –
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = fb.PyTorchModel(model, bounds=(self.clip_min, self.clip_max))

    @sefi_component.attack(name="DeepFool", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        img = torch.from_numpy(img).to(torch.float32).to(self.device)

        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        img = ep.astensor(img)

        # 实例化攻击类
        attack = L2DeepFoolAttack(steps=self.steps, candidates=self.candidates, overshoot=self.overshoot, loss=self.loss)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attack_type == 'UNTARGETED':
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel] ), device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
