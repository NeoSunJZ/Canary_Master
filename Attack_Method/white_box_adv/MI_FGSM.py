import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType
from CANARY_SEFI.evaluator.tester.adv_disturbance_aware import AdvDisturbanceAwareTester

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="MI-FGSM", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="MI-FGSM",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "alpha": {"desc": "攻击算法的学习率(每轮攻击步长)", "type": "FLOAT", "def": "5e-3"},
                                          "epsilon": {"desc": "扰动大小", "type": "FLOAT", "def": "0.01"},
                                          "pixel_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "pixel_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "T": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "1000"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"}})
class MI_FGSM():
    def __init__(self, model, pixel_min, pixel_max, T=1000, epsilon=0.01, alpha=6 / 25, attack_type='UNTARGETED', tlabel=-1):
        self.model = model  # 待攻击的白盒模型
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.pixel_min = pixel_min  # 像素值的下限
        self.pixel_max = pixel_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.alpha = alpha  # 每一轮迭代攻击的步长
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
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

    def attack_iter(self, x, ori_x, sum_grad):
        # 模型梯度清零
        self.model.zero_grad()
        # 模型前向传播
        output = self.model(x)
        # 获取Loss类
        loss_ = self.get_loss()
        # 这里实际上写的不好，如果是其他任务比如人脸做非靶向攻击的话，还得每一轮都找到他的原始人脸特征，再比如如果是靶向标签的话，还需要传入靶向标签。这里只用分类模型非靶向进行测试
        if self.attack_type == 'UNTARGETED':
            loss = loss_(output, torch.Tensor([float(self.label)]).to(self.device).long())  # 非靶向
        else:
            loss = -loss_(output, torch.Tensor([float(self.tlabel)]).to(self.device).long())  # 靶向

        # 反向传播
        loss.backward()
        # MIM梯度累计
        grad = x.grad.data
        grad = grad / torch.std(grad, dim=(1, 2), keepdim=True)
        grad = grad + sum_grad
        sum_grad = grad / torch.std(grad, dim=(1, 2), keepdim=True)
        # 更新图片
        x.data = x.data + self.alpha * torch.sign(sum_grad)
        # clip
        x.data = self.clip_value(x, ori_x)
        return x.data, sum_grad

    @sefi_component.attack(name="MI-FGSM", is_inclass=True, support_model=["vision_transformer"], attack_type="WHITE_BOX")
    def attack(self, img, ori_label):
        self.model.train()
        img = torch.from_numpy(img).to(self.device).float()
        # 定义图片可获取梯度，设置计算图叶子节点
        img.requires_grad = True
        # 克隆原始数据
        ori_x = img.clone()
        # 定义累计梯度
        sum_grad = 0
        # 迭代攻击开始
        if self.attack_type == 'UNTARGETED':
            # self.label = np.argmax(self.model(img).data.to('cpu').numpy())
            self.label = ori_label

        for i in range(self.T):
            img.data, sum_grad = self.attack_iter(img, ori_x, sum_grad)

        img = img.data.cpu().numpy()
        return img

