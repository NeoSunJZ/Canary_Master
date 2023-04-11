import torch

from canary_lib.canary_attack_method.black_box_adv.gen_attack.gen_attack_core import GenAttack as FoolboxGenAttack
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_sefi.handler.tools.foolbox_adapter import FoolboxAdapter

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="GA")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="GA",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
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

        attack = FoolboxGenAttack(steps=step, population=population,
                                  mutation_range=mutation_range, mutation_probability=mutation_probability,
                                  sampling_temperature=sampling_temperature,
                                  channel_axis=1,
                                  reduced_dims=reduced_dims)

        self.foolbox_adapter = FoolboxAdapter(model=model, foolbox_attack=attack,
                                              attack_target=["TARGETED"], required_epsilon=True)

        run_device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.foolbox_adapter.init_args(run_device, attack_type, tlabel, clip_min, clip_max, epsilon)

    @sefi_component.attack(name="GA", is_inclass=True, support_model=[], attack_type="BLACK_BOX")
    def attack(self, imgs, ori_labels, tlabels=None):
        return self.foolbox_adapter.attack(imgs=imgs, ori_labels=ori_labels, target_labels=tlabels)
