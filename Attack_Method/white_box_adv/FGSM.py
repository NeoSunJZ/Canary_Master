import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from foolbox.attacks import LinfFastGradientAttack
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="FGSM", perturbation_budget_var_name="epsilon")
class FGSM():
    def __init__(self, model, T=50, epsilon=0.03, pixel_min=-3, pixel_max=3, alpha=6 / 25, attacktype='untargeted',
                 tlabel=1, dataset = 'imagenet'):
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
        self.dataset = dataset

    # 将图片进行clip
    def clip_value(self, x, ori_x):
        x = torch.clamp((x - ori_x), -self.epsilon, self.epsilon) + ori_x
        x = torch.clamp(x, self.pixel_min, self.pixel_max)
        return x.data

    def get_loss(self):
        return torch.nn.CrossEntropyLoss()

    @sefi_component.attack(name="FGSM", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        # 模型预处理
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        bounds = (-3, 3)
        fmodel = fb.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing)
        #fmodel = fmodel.transform_bounds((-3, 3))

        ori_label = np.array([ori_label])

        # 攻击前需要数据
        img = torch.from_numpy(img).to(torch.float32)
        ori_label = ep.astensor(torch.LongTensor(ori_label))

        img = ep.astensor(img)

        #images, labels = fb.utils.samples(fmodel, dataset=self.dataset, batchsize=16)
        fb.utils.accuracy(fmodel, img, ori_label)

        # 实例化攻击类
        attack = LinfFastGradientAttack()
        self.epsilons = np.linspace(0.0, 0.005, num=20)
        if self.attacktype == 'untargeted':
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon)  # 模型、图像、真标签
            # raw正常攻击产生的对抗样本，clipped通过epsilons剪裁生成的对抗样本，is_adv每个样本的布尔值
            adv_img = raw.raw
        else:
            criterion = TargetedMisclassification(torch.tensor([self.tlabel] ), device=self.device)  # 参数为具有目标类的张量
            raw, clipped, is_adv = attack(fmodel, img, ori_label, epsilons=self.epsilon, criterion=criterion)
            adv_img = raw.raw

        return adv_img  # 正常攻击产生的样本
