import torch
import numpy as np

from foolbox.attacks import BoundaryAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="BA", perturbation_budget_var_name="epsilon")
class BA():
    def __init__(self, model, epsilon=0.03, attacktype='untargeted', tlabel=1, init_attack=None, steps=50,
                 spherical_step=0.01, source_step=0.01, source_step_convergance=1e-07, step_adaptation=1.5,
                 tensorboard=False, update_stats_every_k=10):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.init_attack = init_attack  # 攻击用来寻找起点 仅当starting_points为None时才使用
        self.steps = steps  # 要运行的最大步骤数，可能会在此之前收敛并停止 整型
        self.spherical_step = spherical_step  # 正交（球形）补偿的初始步长
        self.source_step = source_step  # 迈向目标的步骤的初始步长 浮点型
        self.source_step_convergance = source_step_convergance  # 设置停止条件的阈值：如果在攻击期间source_step小于此值，则攻击已收敛并将停止
        self.step_adaptation = step_adaptation  # 步长乘以或除以的因子 浮点型
        self.tensorboard = tensorboard  # TensorBoard摘要的日志目录。如果为False，则TensorBoard摘要将要被禁用；如果为None,则将运行/CURRENT_DATETIME_HOSTNAME
        self.update_stats_every_k =update_stats_every_k  # 整型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @sefi_component.attack(name="BA", is_inclass=True, support_model=["vision_transformer"])
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
        attack = BoundaryAttack(init_attack=self.init_attack, steps=self.steps, spherical_step=self.spherical_step,
                                source_step=self.source_step, source_step_convergance=self.source_step_convergance,
                                step_adaptation=self.step_adaptation, tensorboard=self.tensorboard, update_stats_every_k=self.update_stats_every_k)
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