from functools import reduce

import cv2
import numpy as np
import piq
import torch
from torchvision.transforms import Resize

from CANARY_SEFI.evaluator.tester.frequency_domain_image_processing import get_low_high_f
from CANARY_SEFI.handler.image_handler.img_utils import img_size_uniform_fix


class AdvDisturbanceAwareTester:
    def __init__(self, max_pixel=255.0, min_pixel=0.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel

        self.test_all_handled_img = {}

    def test_all(self, ori_img, adv_img):
        self.test_all_handled_img["ori_img"] = self.img_handler(ori_img)
        self.test_all_handled_img["img"] = self.img_handler(adv_img)
        self.test_all_handled_img["ori_img_forced_int"] = self.img_handler(ori_img, forced_int=True)
        self.test_all_handled_img["img_forced_int"] = self.img_handler(adv_img, forced_int=True)
        high_freq_euclidean_distortion, low_freq_euclidean_distortion = self.calculate_freq_euclidean_distortion(
            ori_img, adv_img)
        result = {
            "maximum_disturbance": float(self.calculate_maximum_disturbance(test_all_model=True)),
            "euclidean_distortion": float(self.calculate_euclidean_distortion(test_all_model=True)),
            "pixel_change_ratio": float(self.calculate_pixel_change_ratio(test_all_model=True)),
            "deep_metrics_similarity": float(self.calculate_deep_metrics_similarity(test_all_model=True)),
            "low_level_metrics_similarity": float(self.calculate_low_level_metrics_similarity(test_all_model=True)),
            "high_freq_euclidean_distortion": high_freq_euclidean_distortion,
            "low_freq_euclidean_distortion": low_freq_euclidean_distortion
        }
        self.test_all_handled_img = {}
        return result


    def calculate_maximum_disturbance(self, ori_img=None, img=None, test_all_model=False):
        if test_all_model:
            ori_img, img = self.test_all_handled_img["ori_img"], self.test_all_handled_img["img"]
        else:
            ori_img, img = self.img_handler(ori_img), self.img_handler(img)
        # L-inf
        result = torch.norm(torch.abs(img - ori_img), float("inf")).cpu().detach().numpy()
        return result

    def calculate_euclidean_distortion(self, ori_img=None, img=None, test_all_model=False):
        if test_all_model:
            ori_img, img = self.test_all_handled_img["ori_img"], self.test_all_handled_img["img"]
        else:
            ori_img, img = self.img_handler(ori_img), self.img_handler(img)
        # L-2
        all_pixel = reduce(lambda x, y: x * y, img.shape)
        result = torch.norm(img - ori_img, 2).cpu().detach().numpy() / all_pixel
        return result

    def calculate_pixel_change_ratio(self, ori_img=None, img=None, test_all_model=False):
        if test_all_model:
            ori_img, img = self.test_all_handled_img["ori_img_forced_int"], self.test_all_handled_img["img_forced_int"]
        else:
            ori_img, img = self.img_handler(ori_img, forced_int=True), self.img_handler(img, forced_int=True)
        # L-0
        all_pixel = reduce(lambda x, y: x * y, img.shape)
        result = torch.norm(img - ori_img, 0).cpu().detach().numpy() / all_pixel
        return result

    def calculate_deep_metrics_similarity(self, ori_img=None, img=None, test_all_model=False):
        if test_all_model:
            ori_img, img = self.test_all_handled_img["ori_img"], self.test_all_handled_img["img"]
        else:
            ori_img, img = self.img_handler(ori_img), self.img_handler(img)

        # DISTS
        result = piq.DISTS(reduction='none')(ori_img.unsqueeze(dim=0), img.unsqueeze(dim=0)).cpu().detach().numpy()
        return result[0]

    def calculate_low_level_metrics_similarity(self, ori_img=None, img=None, test_all_model=False):
        if test_all_model:
            ori_img, img = self.test_all_handled_img["ori_img"], self.test_all_handled_img["img"]
        else:
            ori_img, img = self.img_handler(ori_img), self.img_handler(img)

        # MS-GMSD
        result = piq.MultiScaleGMSDLoss(chromatic=True, data_range=1., reduction='none')\
            (ori_img.unsqueeze(dim=0), img.unsqueeze(dim=0)).cpu().detach().numpy()
        return result[0]

    def calculate_freq_euclidean_distortion(self, ori_img, img):
        radius_ratio = 0.5  # 圆形过滤器的半径：ratio * w/2
        D = 5  # 高斯过滤器的截止频率：2 5 10 20 50 ，越小越模糊信息越少
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        low_freq_part_ori_img, high_freq_part_ori_img = get_low_high_f(ori_img, radius_ratio=radius_ratio, D=D)
        low_freq_part_img, high_freq_part_img = get_low_high_f(img, radius_ratio=radius_ratio, D=D)

        result_low_freq = self.calculate_euclidean_distortion(low_freq_part_ori_img, low_freq_part_img)
        result_high_freq = self.calculate_euclidean_distortion(high_freq_part_ori_img, high_freq_part_img)
        return result_low_freq, result_high_freq

    def img_handler(self, img, forced_int=False):
        if forced_int:
            img = img.copy().astype(np.int)
        img = img / (self.max_pixel - self.min_pixel)
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).to(self.device).float()
