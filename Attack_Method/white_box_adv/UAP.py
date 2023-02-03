import numpy as np
import torch
from torchvision.transforms import Resize
from tqdm import tqdm
from torch.autograd import Variable

from Attack_Method.white_box_adv.deepfool.deepfool import DeepFool
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="UAP")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="UAP",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "pixel_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "pixel_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "num_classes": {"desc": "模型中类的数量", "type": "INT", "required": "true"},
                                          "xi": {"desc": "扰动大小", "type": "FLOAT", "def": "0.01"},
                                          "p": {"desc": "范数类型", "type": "SELECT", "selector": [{"value": "2", "name": "l-2"},{"value": "inf", "name": "l-inf"}], "required": "true"},
                                          "delta": {"desc": "所需误分类率（1-delta）", "type": "FLOAT", "def": "0.2", "required": "true" },
                                          "overshoot": {"desc": "终止条件（防止更新消失）", "type": "FLOAT", "def": "0.02", "required": "true" },
                                          "max_iter_uni": {"desc": "UAP的最大迭代次数", "type": "INT", "def": "100", "required": "true" },
                                          "max_iter_df": {"desc": "DeepFool的最大迭代次数", "type": "INT", "def": "1000", "required": "true" }})
class UAP:
    def __init__(self, model, pixel_min=0, pixel_max=1, num_classes=1000, xi=10 / 255.0, img_size=256, p="l-2", overshoot=0.02, delta=0.2, max_iter_df=1000, max_iter_uni=100):
        self.model = model  # 待攻击的白盒模型
        self.num_classes = num_classes  # 模型中类的数量
        self.img_size = img_size

        self.xi = xi  # 扰动大小
        self.p = p  # 范数类型
        self.delta = delta  # 控制所需愚弄率

        self.max_iter_uni = max_iter_uni
        self.max_iter_df = max_iter_df  # DeepFool的最大迭代次数

        self.overshoot = overshoot # 用作终止条件以防止更新消失
        self.max_iter_df = max_iter_df  # DeepFool的最大迭代次数

        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

        self.v = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    @staticmethod
    def lp(v, xi, p):
        if p == "l-2":
            v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        elif p == "l-inf":
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported.')
        return v

    @sefi_component.attack_init(name="UAP")
    def universal_perturbation(self, dataset):
        print('[ UAP Attack ] p =', self.p, self.xi)
        v = 0
        fooling_rate = 0.0
        iter = 0
        while fooling_rate < 1 - self.delta and iter < self.max_iter_uni:

            # Go through the data set and compute the perturbation increments sequentially
            k = 0
            self.model.to(self.device)
            for cur_img in tqdm(dataset):
                k += 1
                cur_img = torch.from_numpy(cur_img[0]).to(self.device).float()  # 输入img为tensor形式
                resize = Resize([self.img_size,self.img_size])
                cur_img = resize(cur_img)

                per_img = Variable(cur_img + torch.tensor(v).to(self.device), requires_grad = True)
                ori_label = int(self.model(cur_img).argmax())
                per_label = int(self.model(per_img).argmax())
                if ori_label == per_label:
                    self.model.zero_grad()
                    deepfool = DeepFool(self.model,
                                        pixel_min=self.pixel_min,
                                        pixel_max=self.pixel_max,
                                        num_classes=self.num_classes,
                                        overshoot=self.overshoot,
                                        max_iter=self.max_iter_df,
                                        p=self.p)
                    deepfool.attack(per_img.data.cpu().numpy(), None)
                    dr = deepfool.r_tot
                    iter = deepfool.loop_i
                    if iter < self.max_iter_df-1:
                        v = v + dr
                        v = self.lp(v, self.xi, self.p)

            iter = iter + 1

            # Perturb the dataset with computed perturbation
            fooling_sum = 0
            images_sum = 0
            with torch.no_grad():
                for cur_img in tqdm(dataset):
                    cur_img = torch.from_numpy(cur_img[0]).to(self.device).float()  # 输入img为tensor形式
                    resize = Resize([self.img_size, self.img_size])
                    cur_img = resize(cur_img)

                    per_img = cur_img + torch.tensor(v).to(self.device)
                    ori_label = int(self.model(cur_img).argmax())
                    per_label = int(self.model(per_img).argmax())

                    images_sum += 1
                    if ori_label != per_label:
                        fooling_sum += 1

                # Compute the fooling rate
                fooling_rate = fooling_sum / images_sum
                print('FOOLING RATE = ', fooling_rate)
                self.v = v
        return self.v

    @sefi_component.attack(name="UAP", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()
        resize = Resize([self.img_size, self.img_size])
        img = resize(img)

        adv_img = img + torch.tensor(self.v).to(self.device)
        return adv_img.data.cpu().numpy()