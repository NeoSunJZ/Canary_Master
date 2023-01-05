import torch
from foolbox.attacks import EADAttack as FoolboxEADAttack
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from CANARY_SEFI.handler.tools.foolbox_adapter import FoolboxAdapter

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
                                          "binary_search_steps": {"desc": "执行二分搜索时找到扰动范数和分类置信度之间的最佳权衡常数的次数", "type": "INT", "def": "9"},
                                          "steps": {"desc": "更新步骤数", "type": "INT"},
                                          "initial_stepsize": {"desc": "每个二进制搜索步骤中的优化步骤数", "type": "FLOAT","def": "1e-2"},
                                          "confidence": {"desc": "每个二进制搜索步骤中的优化步骤数", "type": "FLOAT", "def": "0.0"},
                                          "initial_const": {"desc": "初始权衡常数，用于调整扰动大小的相对重要性和分类的置信度", "type": "FLOAT", "def": "1e-3"},
                                          "regularization": {"desc": "每个二进制搜索步骤中的优化步骤数", "type": "FLOAT", "def": "1e-2"},
                                          "decision_rule": {"desc": "每个二进制搜索步骤中的优化步骤数", "type": "STRING", "def": "EN"},
                                          "abort_early": {"desc": "每个二进制搜索步骤中的优化步骤数", "type": "BOOLEAN", "def": "True"},
                                      })
class EADAttack:
    def __init__(self, model, run_device, attack_type='UNTARGETED', tlabel=1, clip_min=0, clip_max=1, epsilon=16/255,
                 binary_search_steps=9, steps=10000, initial_stepsize=0.01, confidence=0.0,
                 initial_const=0.001, regularization=0.01, decision_rule='EN', abort_early=True):

        attack = FoolboxEADAttack(binary_search_steps=binary_search_steps, steps=steps,
                                  initial_stepsize=initial_stepsize, confidence=confidence,
                                  initial_const=initial_const, regularization=regularization,
                                  decision_rule=decision_rule, abort_early=abort_early)

        self.foolbox_adapter = FoolboxAdapter(model=model, foolbox_attack=attack,
                                              attack_target=["UNTARGETED", "TARGETED"], required_epsilon=True)

        run_device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.foolbox_adapter.init_args(run_device, attack_type, tlabel, clip_min, clip_max, epsilon)

    @sefi_component.attack(name="EAD", is_inclass=True, support_model=[])
    def attack(self, imgs, ori_labels, tlabels=None):
        return self.foolbox_adapter.attack(imgs=imgs, ori_labels=ori_labels, target_labels=tlabels)
