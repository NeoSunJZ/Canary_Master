from functools import reduce
import piq
import torch


class AdvDisturbanceAwareTester:
    def __init__(self, img_name, max_pixel=255.0, min_pixel=0.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_name = img_name
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel

    def test_all(self, ori_img, img):
        return {
            "maximum_disturbance": self.calculate_maximum_disturbance(ori_img, img),
            "euclidean_distortion": self.calculate_euclidean_distortion(ori_img, img),
            "pixel_change_ratio": self.calculate_pixel_change_ratio(ori_img, img),
            "deep_metrics_similarity": self.calculate_deep_metrics_similarity(ori_img, img),
            "low_level_metrics_similarity": self.calculate_low_level_metrics_similarity(ori_img, img),
        }

    def calculate_maximum_disturbance(self, ori_img, img):
        # L-inf
        result = torch.norm(self.img_handler(img) - self.img_handler(ori_img), float("inf")).cpu().detach().numpy()
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
        # print("计算样本 {} 的 DMS(LPIPS) ".format(self.img_name))
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
        result = piq.MultiScaleGMSDLoss(
            chromatic=True, data_range=1., reduction='none')(x, y).cpu().detach().numpy()
        return result[0]

    def img_handler(self, img):
        img = img / (self.max_pixel - self.min_pixel)
        return torch.from_numpy(img.transpose(2, 0, 1)).to(self.device).float()
