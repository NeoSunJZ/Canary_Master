from functools import reduce

import cv2
import piq
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Resize

from CANARY_SEFI.evaluator.tester.frequency_domain_image_processing import get_low_high_f


class AdvDisturbanceAwareTester:
    def __init__(self, max_pixel=255.0, min_pixel=0.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel

    def test_all(self, ori_img, adv_img):
        ori_img, adv_img = self.img_size_uniform_fix(ori_img, adv_img)
        high_freq_euclidean_distortion, low_freq_euclidean_distortion = self.calculate_freq_euclidean_distortion(
            ori_img, adv_img)
        return {
            "maximum_disturbance": float(self.calculate_maximum_disturbance(ori_img, adv_img)),
            "euclidean_distortion": float(self.calculate_euclidean_distortion(ori_img, adv_img)),
            "pixel_change_ratio": float(self.calculate_pixel_change_ratio(ori_img, adv_img)),
            "deep_metrics_similarity": float(self.calculate_deep_metrics_similarity(ori_img, adv_img)),
            "low_level_metrics_similarity": float(self.calculate_low_level_metrics_similarity(ori_img, adv_img)),
            "high_freq_euclidean_distortion": high_freq_euclidean_distortion,
            "low_freq_euclidean_distortion": low_freq_euclidean_distortion
        }

    def calculate_maximum_disturbance(self, ori_img, img):
        # L-inf
        result = torch.norm(torch.abs(self.img_handler(img) - self.img_handler(ori_img)),
                            float("inf")).cpu().detach().numpy()
        return result

    def calculate_euclidean_distortion(self, ori_img, img):
        # L-2
        img = self.img_handler(img)
        all_pixel = reduce(lambda x, y: x * y, img.shape)
        result = torch.norm(img - self.img_handler(ori_img), 2).cpu().detach().numpy() / all_pixel
        return result

    def calculate_pixel_change_ratio(self, ori_img, img):
        # L-0
        img = self.img_handler(img)
        all_pixel = reduce(lambda x, y: x * y, img.shape)
        result = torch.norm(img - self.img_handler(ori_img), 0).cpu().detach().numpy() / all_pixel
        return result

    def calculate_deep_metrics_similarity(self, ori_img, img):
        # LPIPS
        # lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
        # result = lpips_vgg(self.img_handler(img), self.img_handler(ori_img)).sum().cpu().detach().numpy()

        # DISTS
        x = self.img_handler(ori_img).unsqueeze(dim=0)
        y = self.img_handler(img).unsqueeze(dim=0)
        result = piq.DISTS(reduction='none')(x, y).cpu().detach().numpy()
        return result[0]

    def calculate_low_level_metrics_similarity(self, ori_img, img):
        x = self.img_handler(ori_img).unsqueeze(dim=0)
        y = self.img_handler(img).unsqueeze(dim=0)
        result = piq.MultiScaleGMSDLoss(chromatic=True, data_range=1., reduction='none')(x, y).cpu().detach().numpy()
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

    def img_handler(self, img):
        img = img / (self.max_pixel - self.min_pixel)
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).to(self.device).float()

    def img_size_uniform_fix(self, ori_img, adv_img):
        ori_h, ori_w = ori_img.shape[0], ori_img.shape[1]
        adv_h, adv_w = adv_img.shape[0], adv_img.shape[1]
        if ori_h != adv_h or ori_w != adv_w:
            ori_img = torch.from_numpy(ori_img.transpose(2, 0, 1)).to(self.device).float()
            resize = Resize([adv_h, adv_w])
            ori_img = resize(ori_img)
            ori_img = ori_img.data.cpu().numpy().transpose(1, 2, 0)
        return ori_img, adv_img
