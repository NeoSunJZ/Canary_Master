import torch
import numpy as np

from foolbox.attacks import DDNAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="DDN", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="NewtonFool",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.03"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "steps": {"desc": "更新步骤数", "type": "INT"},
                                          "stepsize": {"desc": "每个更新步骤的大小", "type": "FLOAT"},
                                          "init_epsilon": {"desc": "范数/epsilon ball初始值", "type": "FLOAT"},
                                          "gamma": {"desc": "修改范数的因素：new_norm = norm * (1 + or - gamma)", "type": "FLOAT"},
                                      })
class DDN():
    def __init__(self, model, epsilon=0.03, attack_type='UNTARGETED', tlabel=1, init_epsilon=1.0, steps=100, gamma=0.05,
                 clip_min=0, clip_max=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_epsilon = init_epsilon  # 范数/epsilon ball初始值 浮点型
        self.steps = steps  # 优化步骤数 整型
        self.gamma = gamma  # 修改范数的因素：new_norm = norm * (1 + or - gamma) 浮点型
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = fb.PyTorchModel(model, bounds=(self.clip_min, self.clip_max))

    @sefi_component.attack(name="DDN", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        img = torch.from_numpy(img).to(torch.float32).to(self.device)

        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        img = ep.astensor(img)

        attack = DDNAttack(init_epsilon=self.init_epsilon, steps=self.steps, gamma=self.gamma)
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attack_type == 'UNTARGETED':
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel] ), device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
