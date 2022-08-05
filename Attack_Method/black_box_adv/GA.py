import torch
import numpy as np

from foolbox.attacks import GenAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="GA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="GA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "attack_type": {"desc": "攻击类型(靶向(TARGETED) / 非靶向(UNTARGETED))", "type": "INT"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "steps": { "type": "INT", "df_v": "1000"},
                                          "population": {"type": "INT", "df_v": "10"},
                                          "mutation_probability": {"type": "FLOAT", "df_v": "0.10"},
                                          "mutation_range": {"type": "FLOAT", "df_v": "0.15"},
                                          "sampling_temperature": {"type": "FLOAT", "df_v": "0.3"},
                                          "channel_axis": {"type": "INT", "df_v": "None"},
                                          "reduced_dims": {"type": "Tuple[int,int]", "df_v": "None"}})
class GA():
    def __init__(self, model, epsilon=0.03, attacktype='TARGETED', tlabel=1, steps=1000, population=10, mutation_probability=0.10,
                 mutation_range=0.15, sampling_temperature=0.3, channel_axis=None, reduced_dims=None):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.steps = steps  # 整型
        self.population = population  # 整型
        self.mutation_probability = mutation_probability  # 突变概率 浮点型
        self.mutation_range = mutation_range  # 突变范围 浮点型
        self.sampling_temperature = sampling_temperature  # 采样温度 浮点型
        self.channel_axis = channel_axis  # 整型
        self.reduced_dims = reduced_dims  # 元组[整型，整型]

    @sefi_component.attack(name="GA", is_inclass=True, support_model=["vision_transformer"], attack_type="BLACK_BOX")
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
        attack = GenAttack(steps=self.steps, population=self.population, mutation_probability=self.mutation_probability,
                           mutation_range = self.mutation_range, sampling_temperature=self.sampling_temperature,
                           channel_axis=self.channel_axis, reduced_dims=self.reduced_dims)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel]))  # 参数为具有目标类的张量
        raw, clipped, is_adv = attack(model=fmodel, inputs=img, epsilons=self.epsilon, criterion=criterion)
        # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
