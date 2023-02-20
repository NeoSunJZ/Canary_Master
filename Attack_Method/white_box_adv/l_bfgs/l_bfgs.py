import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import eagerpy as ep

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="L_BFGS")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="L_BFGS",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true", "def": "0.0"},
                                          "clip_max": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true", "def": "1.0"},
                                          "steps": {"desc": "最大迭代次数", "type": "INT", "required": "true", "def": "10"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"}, {"value": "UNTARGETED", "name": "非靶向"}], "required": "true", "def": "UNTARGETED"},
                                          "attack_target": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT", "required": "true", "def": "None"},
                                      })
class L_BFGS():
    def __init__(self, model, run_device, clip_min=0.0, clip_max=1.0, steps=10, attack_type="UNTARGETED", tlabel=None,):
        self._adv = None  # 对抗样本
        self.model = model  # 模型
        self.device = run_device  # 设备
        self.bounds_min = clip_min
        self.bounds_max = clip_max
        self.bounds = clip_min, clip_max
        self.steps = steps
        self.attack_type = attack_type
        self.attack_target = tlabel
        self.epsilon = 0.1
        self._output = None

    def __call__(self, data, target, real_target):
        self.data = data.to(self.device)
        self.target = torch.tensor([target]).to(self.device)
        self.real_target = real_target.to(self.device)
        # 初始化变量c
        c = 1
        x0 = self.data.clone().cpu().numpy().flatten().astype(float)
        # 线搜索算法
        is_adversary = False
        for i in range(30):
            c = 2 * c
            # print('c={}'.format(c))
            is_adversary = self._lbfgsb(x0, c, self.steps)
            if is_adversary:
                # print('扰动成功', c)
                break

        if not is_adversary:
            # print('扰动失败 ')
            return self._adv

        # 使用二分法优化最后的参数
        c_low = 0
        c_high = c
        while c_high - c_low >= self.epsilon:
            c_half = (c_low + c_high) / 2
            old_c = c_half
            is_adversary = self._lbfgsb(x0, c_half, self.steps)
            if is_adversary:
                c_high = c_high - self.epsilon
            else:
                c_low = c_half
                if c_high - c_low <= self.epsilon:
                    # print('出现特殊情况,参数更新时，由扰动成功转化为扰动失败')
                    self._lbfgsb(x0, old_c, self.steps)  # 有可能在优化参数c的过程，参数更新时，由扰动成功转化为扰动失败，且此时已经达到最优点，无法进行更新，需要将上一步扰动成功的进行保存数值
        # print('最优的c:', c_half)

    def _loss(self, adv_x, c):

        adv = torch.from_numpy(adv_x.reshape(self.data.size())).float().to(self.device).requires_grad_(True)

        # 交叉熵损失
        output = self.model(adv)
        ce = F.cross_entropy(output, self.target.long())
        # L2
        d = torch.sum((self.data - adv) ** 2)

        #  Loss
        loss = c * ce + d

        # 梯度
        loss.backward()
        grad_ret = adv.grad.data.cpu().numpy().flatten().astype(float)
        loss = loss.data.cpu().numpy().flatten().astype(float)

        return loss, grad_ret

    def _lbfgsb(self, x0, c, maxiter):
        min_, max_ = self.bounds
        bounds = [(min_, max_)] * len(x0)
        approx_grad_eps = (max_ - min_) / 100.0
        x, f, d = fmin_l_bfgs_b(
            self._loss,
            x0,
            args=(c,),
            bounds=bounds,
            maxiter=maxiter,
            epsilon=approx_grad_eps)
        if np.amax(x) > max_ or np.amin(x) < min_:
            x = np.clip(x, min_, max_)

        adv = torch.from_numpy(x.reshape(self.data.shape)).float().to(self.device)
        output = self.model(adv)
        adv_label = output.max(1, keepdim=True)[1]
        # print('pre_label = {}, adv_label={}'.format(self.real_target, adv_label))

        if self._adv is None:
            self._adv = adv

        self._output = output
        if self.attack_type == "UNTARGETED":
            if self.real_target.item() != adv_label.item():  # 扰动成功，返回值，注意此时为非目标扰动
                self._adv = adv
                return True
            else:
                return False
        elif self.attack_type == "TARGETED":
            if adv_label.item() == self.target.item():  # 使用这个指定为目标扰动
                self._adv = adv
                return True
            else:
                return False
        else:
            raise RuntimeError("[ Logic Error ] Illegal target type!")

    @sefi_component.attack(name="L_BFGS", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, img, ori_label, tlabels=None):
        img_adv = torch.zeros(img.size())
        ori_label = np.array([ori_label])
        ori_label = ep.astensor(torch.LongTensor(ori_label).to(self.device))
        ori_label = ori_label.squeeze(0).numpy()
        length = img.size()[0]
        tlabels = np.repeat(self.attack_target, length) if tlabels is None else tlabels

        for i in range(length):  # 为了能跑batch
            one_img = torch.unsqueeze(img[i], dim=0)
            one_label = torch.tensor(ori_label[i])
            one_tlabel = torch.tensor(tlabels[i])

            self.__call__(one_img, one_tlabel, one_label)
            adv = self._adv
            adv_numpy = adv.detach().cpu().numpy()
            img_adv[i] = torch.from_numpy(adv_numpy)

        img_adv = img_adv.to(self.device)

        return img_adv



