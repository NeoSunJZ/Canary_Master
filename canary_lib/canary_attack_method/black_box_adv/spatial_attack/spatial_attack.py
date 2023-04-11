import torch
from foolbox.attacks import SpatialAttack as FoolBoxSpatialAttack
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_sefi.handler.tools.foolbox_adapter import FoolboxAdapter

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="SpatialAttack", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="SpatialAttack",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "ARRAY_INT"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "max_translation": {"desc": "映射坐标与原坐标最大差值", "type": "FLOAT", "def": "3"},
                                          "max_rotation": {"desc": "映射角度与原坐标最大差值", "type": "FLOAT", "def": "30"},
                                          "num_translations": {"desc": "生成的translation的数量（grid_search为True时按[-max_translation, max_translation]平均生成）", "type": "INT", "def": "5"},
                                          "num_rotations": {"desc": "生成的rotation的数量（grid_search为True时按[-max_rotation, max_rotation]平均生成）", "type": "INT", "def": "5"},
                                          "grid_search": {"desc": "是否按grid均匀生成映射", "type": "BOOLEAN", "def": "TRUE"},
                                          "random_steps": {"desc": "随机生成的映射的数量（仅当grid_search为False时有效）", "type": "INT", "def": "100"}
                                      })
class SpatialAttack:
    def __init__(self, model, run_device, attack_type='UNTARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=None,
                 max_translation=3, max_rotation=30,
                 num_translations=5, num_rotations=5,
                 grid_search=True, random_steps=100):
        attack = FoolBoxSpatialAttack(max_translation=max_translation, max_rotation=max_rotation,
                                      num_translations=num_translations, num_rotations=num_rotations,
                                      grid_search=grid_search, random_steps=random_steps)
        self.foolbox_adapter = FoolboxAdapter(model=model, foolbox_attack=attack,
                                              attack_target=["UNTARGETED", "TARGETED"], required_epsilon=False)

        run_device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.foolbox_adapter.init_args(run_device, attack_type, tlabel, clip_min, clip_max, epsilon)

    @sefi_component.attack(name="SpatialAttack", is_inclass=True, support_model=[])
    def attack(self, imgs, ori_labels, tlabels=None):
        return self.foolbox_adapter.attack(imgs=imgs, ori_labels=ori_labels, target_labels=tlabels)
