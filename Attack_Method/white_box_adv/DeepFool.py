import torch
from torch.autograd import Variable
import copy
import numpy as np
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="DeepFool", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="DeepFool",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "pixel_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "pixel_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "p": {"desc": "范数类型", "type": "SELECT", "selector": [{"value": "2", "name": "l-2"}, {"value": "inf", "name": "l-inf"}], "required": "true"},
                                          "max_iter": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "1000"},
                                          "num_classes": {"desc": "模型中类的数量", "type": "INT", "required": "true"},
                                          "overshoot": {"desc": "最大超出边界的值", "type": "FLOAT", "def": "0.02"},
                                      })
class DeepFool():
    def __init__(self, model, pixel_min=0, pixel_max=1, p="l-2", num_classes=1000, overshoot=0.02, max_iter=50):
        self.model = model  # 待攻击的白盒模型
        self.num_classes = num_classes  # 模型中类的数量
        self.overshoot = overshoot  # 边界超出量，用作终止条件以防止类别更新
        self.max_iter = max_iter # FoolBox的最大迭代次数

        self.p = p

        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 全局变量
        self.label = -1
        self.r_tot = None
        self.loop_i = 0

    def clip_value(self, x):
        x = torch.clamp(x, self.pixel_min, self.pixel_max)
        return x.data

    @sefi_component.attack(name="DeepFool", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, image, ori_label):
        """
           :param image: Image of size 3*H*W
           :param net: network (input: images, output: values of activation **BEFORE** softmax).
           :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
        """
        image = torch.from_numpy(image).to(self.device).float()
        image = Variable(image, requires_grad=True)
        f_image = self.model(image).data.cpu().numpy().flatten()
        I = f_image.argsort()[::-1]  # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
        I = I[0:self.num_classes]
        self.label = I[0]

        input_shape = image.detach().cpu().numpy().shape
        pert_image = copy.deepcopy(image)

        w = np.zeros(input_shape)
        self.r_tot = np.zeros(input_shape)
        self.loop_i = 0

        x = Variable(pert_image, requires_grad=True)
        fs = self.model(x)

        k_i = self.label

        while k_i == self.label and self.loop_i < self.max_iter:
            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):
                x.grad.zero_()
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = self.get_distances(f_k, w_k)

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_i = self.get_perturbations(pert + 1e-4, w)

            self.r_tot = np.float32(self.r_tot + r_i)

            pert_image = image + (1 + self.overshoot) * torch.from_numpy(self.r_tot).to(self.device)

            pert_image = self.clip_value(pert_image)

            x = Variable(pert_image, requires_grad=True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            self.loop_i += 1

        self.r_tot = (1 + self.overshoot) * self.r_tot

        return pert_image.data.cpu().numpy()

    def get_distances(self, losses, grads):
        if self.p == "l-2":
            return abs(losses) / np.linalg.norm(grads.flatten(), ord=2)
        elif self.p == "l-inf":
            return abs(losses) / np.linalg.norm(grads.flatten(), ord=np.inf)
        else:
            raise TypeError("[ Type Error ] Unsupported norm type!")

    def get_perturbations(self, distances, grads):
        if self.p == "l-2":
            return distances * grads / np.linalg.norm(grads.flatten(), ord=2)
        elif self.p == "l-inf":
            return distances * grads / np.linalg.norm(grads.flatten(), ord=np.inf)
        else:
            raise TypeError("[ Type Error ] Unsupported norm type!")