import torch
import numpy as np

from foolbox.attacks import SpatialAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="SA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="SA",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.03"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "max_translation": {"desc": "映射坐标与原坐标最大差值", "type": "FLOAT", "def": "3"},
                                          "max_rotation": {"desc": "映射角度与原坐标最大差值", "type": "FLOAT", "def": "30"},
                                          "num_translations": {"desc": "生成的translation的数量（grid_search为True时按[-max_translation, max_translation]平均生成）", "type": "INT", "def": "5"},
                                          "num_rotations": {"desc": "生成的rotation的数量（grid_search为True时按[-max_rotation, max_rotation]平均生成）", "type": "INT", "def": "5"},
                                          "grid_search": {"desc": "是否按grid均匀生成映射", "type": "BOOLEAN", "def": "TRUE"},
                                          "random_steps": {"desc": "随机生成的映射的数量（仅当grid_search为False时有效）", "type": "INT", "def": "100"}
                                      })
class SA():
    def __init__(self, model, epsilon=0.03, attack_type='UNTARGETED', tlabel=1, max_translation=3, max_rotation=30,
                 num_translations=5, num_rotations=5, grid_search=True, random_steps=100, clip_min=0, clip_max=1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_translation = max_translation  # 浮点型
        self.max_rotation = max_rotation  # 浮点型
        self.num_translations = num_translations  # 整型
        self.num_rotations = num_rotations  # 整型
        self.grid_search = grid_search  # 布尔型
        self.random_steps = random_steps  # 整型
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = fb.PyTorchModel(model, bounds=(self.clip_min, self.clip_max))

    @sefi_component.attack(name="SA", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        img = torch.from_numpy(img).to(torch.float32).to(self.device)

        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        img = ep.astensor(img)

        # 实例化攻击类
        attack = SpatialAttack(max_translation=self.max_translation, max_rotation=self.max_rotation, num_translations=self.num_translations,
                               num_rotations=self.num_rotations, grid_search=self.grid_search, random_steps=self.random_steps)
        if self.attack_type == 'UNTARGETED':
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        else:
            criterion = TargetedMisclassification(target_classes=torch.tensor([self.tlabel] ), device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon, criterion=criterion)

        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量

        return adv_img
