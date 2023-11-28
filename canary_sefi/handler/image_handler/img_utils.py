import math

import numpy as np
import torch
from numpy import average, dot, linalg
from torchvision.transforms import Resize


def img_size_uniform_fix(ori_img, target_img, use_raw_nparray_data):
    img_type = None
    if use_raw_nparray_data:
        img_type = np.float32
    else:
        img_type = int
    ori_h, ori_w = ori_img.shape[0], ori_img.shape[1]
    adv_h, adv_w = target_img.shape[0], target_img.shape[1]
    if ori_h != adv_h or ori_w != adv_w:
        ori_img = ori_img.copy().astype(img_type)
        ori_img = ori_img.transpose(2, 0, 1)
        ori_img = torch.from_numpy(ori_img).to('cuda' if torch.cuda.is_available() else 'cpu').float()
        resize = Resize([adv_h, adv_w])
        ori_img = resize(ori_img)
        ori_img = ori_img.data.cpu().numpy().transpose(1, 2, 0)
        ori_img = np.clip(ori_img, 0, 255).astype(np.float32)
    return ori_img, target_img


def get_img_diff(original_img, adversarial_img):
    difference = adversarial_img - original_img
    difference = difference / abs(difference).max() / 2.0 + 0.5
    return difference


def get_img_cosine_similarity(img1, img2):
    if img1 is None or img2 is None:
        return None
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
    if math.isnan(res):
        return None
    return res
