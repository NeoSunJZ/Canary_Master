import torch

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="CW", perturbation_budget_var_name="epsilon")
class CW():
    def __init__(self, model, classes=1000, lr=6/25, confidence=0, clip_min=-3, clip_max=3,initial_const=1e-2,
                 binary_search_steps=5, max_iterations=50,attacktype='untargeted', tlabel=1, epsilon=0.2):
        self.model = model  # 待攻击的白盒模型
        self.n_classes = classes  # 模型中类的数量
        self.lr = lr  # 攻击算法的学习速率（浮点数）
        self.confidence = confidence  # 对抗性示例的置信度：越高，生成的示例的l2失真越大，但更强烈地归类为对抗性示例 默认为0
        self.clip_min = clip_min  # 对抗性示例组件的最小浮点值
        self.clip_max = clip_max  # 对抗性示例组件的最大浮点值
        self.initial_const = initial_const  # 初始权衡常量，用于调整扰动大小的相对重要性和分类的置信度
        self.binary_search_steps = binary_search_steps  # 执行二进制搜索已找到扰动范数和分类置信度之间的最佳权衡常数的次数 整型
        self.max_iterations = max_iterations  # 最大迭代次数 整型
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel  # 靶向攻击目标标签
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值


    @sefi_component.attack(name="CW", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()  # 输入img为temsor形式
        y = torch.tensor([self.tlabel],device=self.device)  # 具有真实标签的张量 默认为None

        if self.attacktype == 'untargeted':
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
        else:
            img = carlini_wagner_l2(model_fn=self.model,
                                    x=img,
                                    n_classes=self.n_classes,
                                    y=y,
                                    targeted=True,
                                    lr=self.lr,
                                    confidence=self.confidence,
                                    clip_min=self.clip_min,
                                    clip_max=self.clip_max,
                                    initial_const=self.initial_const,
                                    binary_search_steps=self.binary_search_steps,
                                    max_iterations=self.max_iterations)

        return img
