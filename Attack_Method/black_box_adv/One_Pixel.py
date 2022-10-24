import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType
from .one_pixel.differential_evolution import differential_evolution
from torch.autograd import Variable
import torch.nn.functional as F


sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="One_Pixel", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="One_Pixel",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "epsilon": {"desc": "扰动大小", "type": "FLOAT", "def": "0.2"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT", "selector": [{"value": "TARGETED", "name": "靶向"}, {"value": "UNTARGETED", "name": "非靶向"}], "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "population": {"desc": "初始权衡常数，用于调整扰动大小的相对重要性和分类的置信度", "type": "INT", "def": "1e-2"},
                                          "max_iter": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "100"},
                                          "pixels": {"desc": "改变像素的数量", "type": "INT", "def": "1"},
                                      })
class OnePixel():
    def __init__(self, model, max_iter=100, epsilon=0.2, clip_min=-3, clip_max=3, pixels=1, population=400, attack_type='UNTARGETED',
                 tlabel=-1):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限（这与当前图片范围有一定关系，建议0-255，因为对于无穷约束来将不会因为clip原因有一定损失）
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.population = population
        self.max_iter = max_iter
        self.pixels = pixels

    @sefi_component.attack(name="One_Pixel", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, img, ori_label):
        _, c, h, w, = img.shape
        img = torch.from_numpy(img).to(self.device).float()

        bounds = [(0, h), (0, w)]
        bounds += [(self.clip_min, self.clip_max)] * c
        bounds = bounds * self.pixels
        len_bounds = len(bounds)
        popmul = max(1, self.population / len_bounds)


        if self.attack_type == 'UNTARGETED':
            predict_fn = lambda xs: _predict_classes(
                xs, img, ori_label, self.model, False)
            callback_fn = lambda x, convergence: _attack_success(
                x, img, ori_label, self.model, False, False)
        else:
            predict_fn = lambda xs: _predict_classes(
                xs, img, self.tlabel, self.model, True)
            callback_fn = lambda x, convergence: _attack_success(
                x, img, self.tlabel, self.model, True, False)

        inits = np.zeros([int(popmul * len(bounds)), len(bounds)])
        for init in inits:
            for i in range(self.pixels):
                init[i * len_bounds + 0] = np.random.random() * h
                init[i * len_bounds + 1] = np.random.random() * w
                for j in range(c):
                    init[i * len_bounds + 2 + j] = np.random.random() * (self.clip_max - self.clip_min) + self.clip_min

        attack_result = differential_evolution(predict_fn, bounds, maxiter=self.max_iter, popsize=popmul,
                                               recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

        attack_image = _perturb_image(attack_result.x, img)
        with torch.no_grad():
            attack_var = Variable(attack_image).cuda()
            predicted_probs = F.softmax(self.model(attack_var), dim=1).data.cpu().numpy()[0]

        predicted_class = np.argmax(predicted_probs)

        if (self.attack_type == 'UNTARGETED' and predicted_class != ori_label) or (self.attack_type != 'UNTARGETED' and predicted_class == self.tlabel):
            return attack_image.cpu().detach().numpy()

        return img.cpu().detach().numpy()


def _perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x) / 5)

        for pixel in pixels:
            # 核心代码这里
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = r
            imgs[count, 1, x_pos, y_pos] = g
            imgs[count, 2, x_pos, y_pos] = b
        count += 1

    return imgs


def _predict_classes(xs, img, target_class, net, minimize=True):
    imgs_perturbed = _perturb_image(xs, img.clone())
    with torch.no_grad():
        input = Variable(imgs_perturbed).cuda()
        predictions = F.softmax(net(input), dim=1).data.cpu().numpy()[:, target_class]

    return predictions if minimize else 1 - predictions


def _attack_success(x, img, target_class, net, targeted_attack=False, verbose=False):

    attack_image = _perturb_image(x, img.clone())
    with torch.no_grad():
        input = Variable(attack_image).cuda()
        confidence = F.softmax(net(input), dim=1).data.cpu().numpy()[0]
        predicted_class = np.argmax(confidence)

    if (verbose):
        print("Confidence: %.4f" % confidence[target_class])
    if (targeted_attack and predicted_class == target_class) or (
            not targeted_attack and predicted_class != target_class and confidence[target_class] < 0.2):
        return True