import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType
from .one_pixel.differential_evolution import differential_evolution
from torch.autograd import Variable
import torch.nn.functional as F
from .zoo.ZOOOptimizer import ZOOptim
from .zoo.ZOOLossFunction import ZooLoss

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="ZOO", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="ZOO",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "epsilon": {"desc": "扰动大小", "type": "FLOAT", "def": "0.2"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"},{"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "max_iter": {"desc": "最大迭代数", "type": "INT", "def": "10000"},
                                          "learning_rate": {"desc": "优化器学习率", "type": "FLOAT", "def": "2e-3"},
                                          "solver": {"desc": "模拟梯度所使用的算法", "type": "SELECT", "selector": [{"value": "adam", "name": "adam"}, {"value": "newton", "name": "newton"}], "required": "true"},
                                          "loss_weight": {"desc": "最小化问题中的超参(扰动大小 + c * 置信损失)", "type": "FLOAT", "def": "0.3"},
                                          "stop_criterion": {"desc": "是否提前终止（20轮loss无下降）", "type": "BOOLEAN"},
                                          "n_gradient": {"desc": "同时优化的坐标数", "type": "INT"},
                                          "h": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "beta_1": {"desc": "ADAM超参", "type": "FLOAT", "def": "0.9"},
                                          "beta_2": {"desc": "ADAM超参", "type": "FLOAT", "def": "0.999"},
                                      })
class ZOO():
    def __init__(self, model, max_iter=10000, epsilon=0.3, clip_min=-3, clip_max=3, attack_type='UNTARGETED',
                 tlabel=-1, learning_rate=2e-3, solver="adam", loss_weight=0.3, stop_criterion=True,
                 h=1e-4, beta_1=0.9, beta_2=0.999, n_gradient=128):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver # "adam" or "newton"
        self.loss_weight = loss_weight
        self.stop_criterion = stop_criterion
        self.n_gradient = n_gradient
        self.h = h
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @sefi_component.attack(name="ZOO", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        img = torch.from_numpy(img).to(self.device).float()
        img = torch.squeeze(img, 0)
        if self.attack_type == "UNTARGETED":
            ### Define our custom loss function and call the optimizer
            adv_loss = ZooLoss(neuron=ori_label, maximise=0, is_softmax=False, dim=1)
            adv_optimizer = ZOOptim(model=self.model, loss=adv_loss, device=self.device)

            ### Run the optimizer
            x, loss, outs = adv_optimizer.run(img, c=self.loss_weight,
                                                                  learning_rate=self.learning_rate,
                                                                  n_gradient=self.n_gradient, h=self.h,
                                                                  beta_1=self.beta_1,
                                                                  beta_2=self.beta_2, solver=self.solver, verbose=False,
                                                                  max_steps=self.max_iter, batch_size=-1,
                                                                  C=(self.clip_min, self.clip_max),
                                                                  stop_criterion=self.stop_criterion,
                                                                  tqdm_disabled=False, additional_out=False)
        else:
            ### Define our custom loss function and call the optimizer
            adv_loss = ZooLoss(neuron=self.tlabel, maximise=1, is_softmax=False, dim=1)
            adv_optimizer = ZOOptim(model=self.model, loss=adv_loss, device=self.device)

            ### Run the optimizer
            x, loss, outs = adv_optimizer.run(img, c=self.loss_weight,
                                                                  learning_rate=self.learning_rate,
                                                                  n_gradient=self.n_gradient, h=self.h,
                                                                  beta_1=self.beta_1,
                                                                  beta_2=self.beta_2, solver=self.solver, verbose=False,
                                                                  max_steps=self.max_iter, batch_size=-1,
                                                                  C=(self.clip_min, self.clip_max),
                                                                  stop_criterion=self.stop_criterion,
                                                                  tqdm_disabled=False, additional_out=False)

        return np.expand_dims(x.cpu().detach().numpy(), axis=0)