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
                                          "attack_type": {"desc": "攻击类型(靶向(TARGETED) / 非靶向(UNTARGETED))", "type": "INT"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "steps": {"desc": "要执行的最大步骤数", "type": "INT", "df_v": "50"},
                                          "candidates": {"desc": "限制应考虑的最可能类的数量", "type": "INT", "df_v": "10"},
                                          "overshoot": {"desc": "超出边界的量", "type": "FLOAT", "df_v": "0.02"},
                                          "loss": {"desc": "Union[typing_extensions.Literal['logits'], typing_extensions.Literal['crossentropy']]", "df_v": "'logits'"}})
class DeepFool():
    def __init__(self, model, epsilon=0.03, attacktype='untargeted', tlabel=1, steps=50, candidates=10,
                 overshoot=0.02, loss='logits'):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.steps = steps  # 要执行的最大步骤数
        self.candidates = candidates  # 限制应考虑的最可能类的数量
        self.overshoot = overshoot  # 超出边界的量 浮点型
        self.loss = loss  # (Union[typing_extensions.Literal['logits'], typing_extensions.Literal['crossentropy']]) –

    @sefi_component.attack(name="DeepFool", is_inclass=True, support_model=["vision_transformer"])
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
        attack = L2DeepFoolAttack(steps=self.steps, candidates=self.candidates, overshoot=self.overshoot, loss=self.loss)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attacktype == 'untargeted':
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel] ), device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
