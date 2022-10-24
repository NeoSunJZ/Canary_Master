import torch
import numpy as np

from foolbox.attacks import HopSkipJumpAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="HSJA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="HSJA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.03"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "steps": {"desc": "循环次数", "type": "INT", "def": "64"},
                                          "initial_gradient_eval_steps": {"desc": "用于估计梯度方向的向量数量", "type": "INT", "def": "100"},
                                          "max_gradient_eval_steps": {"desc": "估计梯度时最大的估计次数", "type": "INT", "def": "10000"},
                                          "gamma": {"desc": "设置二分搜索停止条件的阈值（如果二分两点的距离小于此值，则认为已经搜索到边界）", "type": "FLOAT", "def": "1.0"},
                                          "constraint": {"desc": "范数", "type": "SELECT", "selector": [{"value": "l2", "name": "l2范数"},{"value": "linf", "name": "l∞范数"}], "required": "true"}
                                      })
class HSJA():
    def __init__(self, model, epsilon=0.03, attack_type='UNTARGETED', tlabel=1, init_attack=None, steps=64,
                 initial_gradient_eval_steps=100, max_gradient_eval_steps=10000, gamma: float = 1.0, constraint="l2",
                 tensorboard=False, clip_min=0, clip_max=1):
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.init_attack = init_attack  # 攻击用来寻找起点
        self.steps = steps  # 二叉搜索最大步数（让xt到bound）
        self.initial_gradient_eval_steps = initial_gradient_eval_steps  # Initial number of evaluations for gradient estimation 方法用蒙特卡洛模拟梯度（多个方向的点用于计算梯度）
        self.max_gradient_eval_steps = max_gradient_eval_steps  # Maximum number of evaluations for gradient estimation.
        self.gamma = gamma  # 设置二分搜索停止条件的阈值：如果二分两点的距离小于此值，则认为已经搜索到边界
        self.constraint = constraint  # 范数
        self.tensorboard = tensorboard  # TensorBoard摘要的日志目录。如果为False，则TensorBoard摘要将要被禁用；如果为None,则将运行/CURRENT_DATETIME_HOSTNAME
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = fb.PyTorchModel(model, bounds=(self.clip_min, self.clip_max))

    @sefi_component.attack(name="HSJA", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        img = torch.from_numpy(img).to(torch.float32).to(self.device)

        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        img = ep.astensor(img)

        # 实例化攻击类
        attack = HopSkipJumpAttack(init_attack=self.init_attack, steps=self.steps,
                                   initial_gradient_eval_steps=self.initial_gradient_eval_steps,
                                   max_gradient_eval_steps=self.max_gradient_eval_steps,
                                   gamma=self.gamma, tensorboard=self.tensorboard, constraint=self.constraint)
        if self.attack_type == 'UNTARGETED':
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=ep.astensor(torch.LongTensor([self.tlabel]).to(self.device)))  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, img, criterion, epsilons=self.epsilon)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img.cpu().detach().numpy()
