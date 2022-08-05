import numpy as np
import torch

from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="HSJA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="HSJA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "norm": {"desc": "要优化的距离(np.inf/2)"},
                                          "y_target": {"desc": "目标标签的形状张量(仅TARGETED时有效)", "type": "TENSOR", "df_v": "None"},
                                          "image_target": {"desc": "作为初始目标图像的形状张量(仅TARGETED时有效)", "type": "TENSOR", "df_v": "None"},
                                          "initial_num_evals": {"desc": "梯度估计的初始评估次数", "df_v": "100"},
                                          "max_num_evals": {"desc": "梯度估计的最大评估次数", "df_v": "10000"},
                                          "stepsize_search": {"desc": "如何搜索步长(geometric_progression/grid_search)", "type": "STR", "df_v": "geometric_progression"},
                                          "num_iterations": {"desc": "迭代次数", "df_v": "64"},
                                          "gamma": {"desc": "二进制搜索阈值(l2攻击(gamma/d^{3/2})/linf攻击(gamma/d^2)", "df_v": "1.0"},
                                          "constraint": {"df_v": "2"},
                                          "batch_size": {"desc": "模型预测的batch_size", "df_v": "128"},
                                          "verbose": {"desc": "是否打印每个步骤的距离", "type": "BOOL", "df_v": "True"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "df_v": "0"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "df_v": "1"},
                                          "attack_type": {"desc": "攻击类型(靶向(TARGETED) / 非靶向(UNTARGETED))", "type": "INT"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class HSJA():
    def __init__(self, model, norm=np.inf, y_target=None, image_target=None, initial_num_evals=100, max_num_evals=10000,
                 stepsize_search="geometric_progression", num_iterations=64, gamma=1.0, constraint=2, batch_size=128,
                 verbose=True, clip_min=-3, clip_max=3, attacktype='UNTARGETED', tlabel=1, epsilon=0.2):
        self.model = model  # 待攻击的白盒模型
        self.norm = norm  # 要优化的距离 可能的值：2或np.inf
        self.y_target = y_target  # 目标标签的形状张量(仅TARGETED时有效)
        self.image_target = image_target  # 作为初始目标图像的形状张量(仅TARGETED时有效)
        self.initial_num_evals = initial_num_evals  # 梯度估计的初始评估次数
        self.max_mun_evals = max_num_evals  # 梯度估计的最大评估次数
        self.stepsize_search = stepsize_search  # 如何搜索步长 'geometric_progression','grid_search'
        # 'geometric_progression'步长初始化为 ||x_t - x||_p / sqrt(iteration)，不断减少一半，直到到达边界的目标侧
        # 'grid_search'在网格上选择最优的epsilon，在||x_t - x||_p规模上
        self.num_iterations = num_iterations  # 迭代次数
        self.gamma = gamma  # 二进制搜索阈值  gamma/d^{3/2}用于l2攻击 gamma/d^2用于linf攻击
        self.constraint = constraint
        self.batch_size = batch_size  # 模型预测的batch_size
        self.verbose = verbose  # 是否打印每个步骤的距离 布尔值
        self.clip_min = clip_min  # 最小输入元件值 可选浮点数
        self.clip_max = clip_max  # 最大输入元件值 可选浮点数
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel  # 靶向攻击目标标签
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值


    @sefi_component.attack(name="HSJA", is_inclass=True, support_model=["vision_transformer"], attack_type="BLACK_BOX")
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()  # 输入img为tensor形式
        y = torch.tensor([self.tlabel],device=self.device)  # 具有真实标签的张量 默认为None

        if self.attacktype == 'UNTARGETED':
            img = hop_skip_jump_attack(model_fn=self.model,
                                       x=img,
                                       norm=self.norm,
                                       y_target=self.y_target,
                                       image_target=self.image_target,
                                       initial_num_evals=self.initial_num_evals,
                                       max_num_evals=self.initial_num_evals,
                                       stepsize_search=self.stepsize_search,
                                       num_iterations=self.num_iterations,
                                       gamma=self.gamma,
                                       constraint=self.constraint,
                                       batch_size=self.batch_size,
                                       verbose=self.verbose,
                                       clip_min=self.clip_min,
                                       clip_max=self.clip_max)
        else:
            img = hop_skip_jump_attack(model_fn=self.model,
                                       x=img,
                                       norm=self.norm,
                                       y_target=y,
                                       image_target=self.image_target,
                                       initial_num_evals=self.initial_num_evals,
                                       max_num_evals=self.initial_num_evals,
                                       stepsize_search=self.stepsize_search,
                                       num_iterations=self.num_iterations,
                                       gamma=self.gamma,
                                       constraint=self.constraint,
                                       batch_size=self.batch_size,
                                       verbose=self.verbose,
                                       clip_min=self.clip_min,
                                       clip_max=self.clip_max)

        return img
