import torch

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType

from canary_lib.canary_attack_method.black_box_adv.zoo.zoo_optimizer import ZOOptim
from canary_lib.canary_attack_method.black_box_adv.zoo.zoo_loss_function import ZooLoss


sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="ZOO", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="ZOO",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
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
                                          "mini_batch": {"desc": "最大同时预测数", "type": "INT", "def": "16"},
                                      })
class ZOO():
    def __init__(self, model, run_device, max_iter=10000, epsilon=0.3, clip_min=-3, clip_max=3, attack_type='UNTARGETED',
                 tlabel=-1, learning_rate=2e-3, solver="adam", loss_weight=0.3, stop_criterion=True,
                 h=1e-4, beta_1=0.9, beta_2=0.999, n_gradient=128, mini_batch=-1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver # "adam" or "newton"
        self.loss_weight = loss_weight
        self.stop_criterion = stop_criterion
        self.n_gradient = n_gradient
        self.h = h
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.batch_size = mini_batch

    @sefi_component.attack(name="ZOO", is_inclass=True, support_model=[])
    def attack(self, img, ori_labels):
        # img = torch.from_numpy(img).to(self.device).float()
        img = torch.squeeze(img, 0)
        if self.attack_type == "UNTARGETED":
            ### Define our custom loss function and call the optimizer
            adv_loss = ZooLoss(neuron=ori_labels[0], maximise=0, is_softmax=False, dim=1)
            adv_optimizer = ZOOptim(model=self.model, loss=adv_loss, device=self.device)

            ### Run the optimizer
            x, loss, outs = adv_optimizer.run(img, c=self.loss_weight,
                                                                  learning_rate=self.learning_rate,
                                                                  n_gradient=self.n_gradient, h=self.h,
                                                                  beta_1=self.beta_1,
                                                                  beta_2=self.beta_2, solver=self.solver, verbose=False,
                                                                  max_steps=self.max_iter, batch_size=self.batch_size,
                                                                  C=(self.clip_min, self.clip_max),
                                                                  stop_criterion=self.stop_criterion,
                                                                  tqdm_disabled=True, additional_out=False)
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
                                                                  max_steps=self.max_iter, batch_size=self.batch_size,
                                                                  C=(self.clip_min, self.clip_max),
                                                                  stop_criterion=self.stop_criterion,
                                                                  tqdm_disabled=True, additional_out=False)

        return torch.unsqueeze(x, 0)
