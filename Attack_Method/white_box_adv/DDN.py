import torch
import numpy as np

from foolbox.attacks import DDNAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="DDN", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="DDN",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "attack_type": {"desc": "攻击类型(靶向(TARGETED) / 非靶向(UNTARGETED))", "type": "INT"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "init_epsilon": {"desc": "范数(epsilon ball)初始值", "type": "FLOAT", "df_v": "1.0"},
                                          "steps": {"desc": "优化步骤数", "type": "INT", "df_v": "100"},
                                          "gamma": {"desc": "修改范数的因素(new_norm = norm * (1 + or - gamma))", "type": "FLOAT", "df_v": "0.05"}})

class DDN():
    def __init__(self, model, epsilon=0.03, attacktype='UNTARGETED', tlabel=1, init_epsilon=1.0, steps=100, gamma=0.05):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型(靶向(TARGETED) / 非靶向(UNTARGETED)
        self.tlabel = tlabel  # 靶向攻击目标标签(分类标签)(仅TARGETED时有效)
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_epsilon = init_epsilon  # 范数(epsilon ball)初始值
        self.steps = steps  # 优化步骤数
        self.gamma = gamma  # 修改范数的因素(new_norm = norm * (1 + or - gamma))

    @sefi_component.attack(name="DDN", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
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
        attack = DDNAttack(init_epsilon=self.init_epsilon, steps=self.steps, gamma=self.gamma)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attacktype == 'UNTARGETED':
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel]))  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
