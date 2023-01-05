import torch
import numpy as np

import foolbox as fb
import eagerpy as ep
from Attack_Method.black_box_adv.genattack.GenAttack import GenAttack

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from foolbox.criteria import TargetedMisclassification

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="GA")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="GA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.25"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "step": {"desc": "算法迭代轮数","type": "INT","def":"1000"},
                                          "population": {"desc": "遗传算法人口数，这里是随机选择的一定距离范围内的样本数","type": "INT","def":"10"},
                                          "mutation_probability": {"desc": "变异概率，基于遗传算法","type": "FLOAT","def": "0.10"},
                                          "mutation_range": {"desc": "变异范围，基于遗传算法","type": "FLOAT","def": "0.15"},
                                          "sampling_temperature": {"desc": "采样温度，用于计算遗传算法中的选择概率","type": "FLOAT","def": "0.3"},
                                          "reduced_dims": {"desc": "是否缩减维度","type": "TUPLE[INT, INT]","def": "None"}
                                          })
class GenAttack:
    def __init__(self, model, run_device, attack_type='TARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=None,
                 step=1000, population=10, mutation_probability=0.1, mutation_range=0.15, sampling_temperature=0.3,
                 reduced_dims=None):
        self.model = fb.PyTorchModel(model.to(run_device), bounds=(clip_min, clip_max), device=run_device)
        self.device = run_device
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.tlabel = tlabel
        self.step = step
        self.population = population
        self.mutation_probability = mutation_probability
        self.mutation_range = mutation_range
        self.sampling_temperature =sampling_temperature
        self.channel_axis = 1
        self.reduced_dims = reduced_dims


    @sefi_component.attack(name="GenAttack", is_inclass=True, support_model=[], attack_type="BLACK_BOX")
    def attack(self, imgs, ori_labels, tlabels=None):
        batch_size = imgs.shape[0]
        tlabels = np.repeat(self.tlabel, batch_size) if tlabels is None else tlabels

        # 转为PyTorch变量
        tlabels = ep.astensor(torch.from_numpy(np.array(tlabels)).to(self.device))
        ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        imgs = ep.astensor(imgs)

        # 实例化攻击类
        attack = GenAttack(steps=self.step,
                           population=self.population,
                           mutation_range=self.mutation_range,
                           mutation_probability=self.mutation_probability,
                           sampling_temperature=self.sampling_temperature,
                           channel_axis=self.channel_axis,
                           reduced_dims=self.reduced_dims)
        criterion = TargetedMisclassification(target_classes=torch.tensor(tlabels).to(self.device))  # 参数为具有目标类的张量

        # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值

        # 由EagerPy张量转化为Native张量
        if self.epsilon is None:
            raw, clipped, isadv = attack(self.model, imgs, criterion, epsilons=16/255)
            adv_img = raw.raw
        else:
            raw, clipped, isadv = attack(self.model, imgs, criterion, epsilons=self.epsilon)
            adv_img = clipped.raw
        return adv_img
