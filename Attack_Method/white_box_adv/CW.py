import torch

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="CW")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="CW",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "classes": {"desc": "模型中类的数量", "type": "INT", "required": "true"},
                                          "lr": {"desc": "攻击算法的学习率", "type": "FLOAT", "def": "5e-3"},
                                          "confidence": {"desc": "对抗性示例的置信度，越高则生成对抗样本的l2越大，但误分类置信度越高", "type": "INT", "def": "0"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "initial_const": {"desc": "初始权衡常数，用于调整扰动大小的相对重要性和分类的置信度", "type": "FLOAT", "def": "1e-2"},
                                          "binary_search_steps": {"desc": "执行二分搜索时找到扰动范数和分类置信度之间的最佳权衡常数的次数", "type": "INT", "def": "5"},
                                          "max_iterations": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "1000"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class CW:
    def __init__(self, model, classes=1000, lr=5e-3, confidence=0, clip_min=0, clip_max=1, initial_const=1e-2,
                 binary_search_steps=5, max_iterations=1000,attack_type='UNTARGETED', tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.n_classes = classes  # 模型中类的数量
        self.lr = lr  # 攻击算法的学习速率（浮点数）
        self.confidence = confidence  # 对抗性示例的置信度：越高，生成的示例的l2失真越大，但更强烈地归类为对抗性示例 默认为0
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.initial_const = initial_const  # 初始权衡常量，用于调整扰动大小的相对重要性和分类的置信度
        self.binary_search_steps = binary_search_steps  # 执行二进制搜索已找到扰动范数和分类置信度之间的最佳权衡常数的次数 整型
        self.max_iterations = max_iterations  # 最大迭代次数 整型
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel  # 靶向攻击目标标签
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    @sefi_component.attack(name="CW", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()  # 输入img为tensor形式

        if self.attack_type == 'UNTARGETED':
            img = carlini_wagner_l2(model_fn=self.model,
                                    x=img,
                                    n_classes=self.n_classes,
                                    y=None,
                                    targeted=False,
                                    lr=self.lr,
                                    confidence=self.confidence,
                                    clip_min=self.clip_min,
                                    clip_max=self.clip_max,
                                    initial_const=self.initial_const,
                                    binary_search_steps=self.binary_search_steps,
                                    max_iterations=self.max_iterations)
        elif self.attack_type == 'TARGETED':
            img = carlini_wagner_l2(model_fn=self.model,
                                    x=img,
                                    n_classes=self.n_classes,
                                    y=torch.tensor([self.tlabel],device=self.device),
                                    targeted=True,
                                    lr=self.lr,
                                    confidence=self.confidence,
                                    clip_min=self.clip_min,
                                    clip_max=self.clip_max,
                                    initial_const=self.initial_const,
                                    binary_search_steps=self.binary_search_steps,
                                    max_iterations=self.max_iterations)
        else:
            raise Exception("未知攻击方式")

        return img
