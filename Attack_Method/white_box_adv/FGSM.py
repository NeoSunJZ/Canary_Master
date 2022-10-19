import torch
import numpy as np

from foolbox.attacks import LinfFastGradientAttack
import foolbox as fb
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="FGSM")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="FGSM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "epsilon": {"desc": "对抗样本与原始输入图片的最大变化", "type": "FLOAT", "def": "0.25"},
                                          "random_start": {"desc": "是否允许在ε球的空间范围初始化", "type": "BOOLEAN", "def": "False"},
                                          })
class FGSM():
    def __init__(self, model, clip_min=0, clip_max=1, epsilon=0.25, random_start=False):
        self.model = fb.PyTorchModel(model, bounds=(clip_min, clip_max))  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.random_start = random_start  # 控制是否在允许的epsilon ball内随机启动
        self.clip_min = clip_min
        self.clip_max = clip_max

    @sefi_component.attack(name="FGSM", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        ori_label = np.array([ori_label])
        img = torch.from_numpy(img).to(torch.float32).to(self.device)

        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        img = ep.astensor(img)
        # 实例化攻击类
        attack = LinfFastGradientAttack(random_start=self.random_start)
        raw, clipped, is_adv = attack(self.model, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
        # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
        adv_img = raw.raw
        # 由EagerPy张量转化为Native张量
        return adv_img.cpu().numpy()
