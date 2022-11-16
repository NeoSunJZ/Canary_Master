import numpy as np
import torch
from numpy import average, dot, linalg
from torchvision.transforms import Resize


def img_size_uniform_fix(ori_img, target_img):
    ori_h, ori_w = ori_img.shape[0], ori_img.shape[1]
    adv_h, adv_w = target_img.shape[0], target_img.shape[1]
    if ori_h != adv_h or ori_w != adv_w:
        ori_img = torch.from_numpy(ori_img.transpose(2, 0, 1)).cpu()
        resize = Resize([adv_h, adv_w])
        ori_img = resize(ori_img)
        ori_img = ori_img.data.cpu().numpy().transpose(1, 2, 0)
    return ori_img, target_img


def get_img_diff(original_img, adversarial_img):
    difference = adversarial_img - original_img
    difference = difference / abs(difference).max() / 2.0 + 0.5
    return difference


def get_img_cosine_similarity(img1, img2):
    imgs = [img1, img2]
    vectors = []
    norms = []
    for img in imgs:
        vector = []
        for pixel_tuple in img:
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res