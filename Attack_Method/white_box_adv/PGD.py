import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="PGD", perturbation_budget_var_name="epsilon")
class PGD():
    def __init__(self, model, T=50, epsilon=0.2, pixel_min=-3, pixel_max=3, alpha=0.1, attacktype='untargeted',
                 tlabel=1):
        self.model = model  # 待攻击的白盒模型
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.pixel_min = pixel_min  # 像素值的下限
        self.pixel_max = pixel_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.alpha = alpha  # 每一轮迭代攻击的步长
        self.attacktype = attacktype  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.label = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 将图片进行clip
    def clip_value(self, x, ori_x):
        x = torch.clamp((x - ori_x), -self.epsilon, self.epsilon) + ori_x
        x = torch.clamp(x, self.pixel_min, self.pixel_max)
        return x.data

    def get_loss(self):
        return torch.nn.CrossEntropyLoss()

    @sefi_component.attack(name="PGD", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        batch_size = 16
        img = torch.from_numpy(img).to(self.device).float()
        #tlabel_tensor = torch.Tensor([float(self.tlabel)])
        y = torch.tensor([self.tlabel]*batch_size,device=self.device)

        if self.attacktype == 'untargeted':
            img = projected_gradient_descent(self.model, img, self.epsilon, self.alpha, self.T, np.inf, targeted=False) #非靶向 n_classes为int类型
            #model_fn, x, eps, eps_iter, nb_iter, norm, targeted,y
            #projected_gradient_descent中 'assert eps_iter <= eps, (eps_iter, eps)'
        else:
            img = projected_gradient_descent(self.model, img, self.epsilon, self.alpha, self.T, np.inf, targeted=True, y=y) #靶向 y带有真标签的张量

        return img
