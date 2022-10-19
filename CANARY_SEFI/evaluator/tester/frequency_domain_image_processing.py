# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_filter_high_f(fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    template = np.exp(- dis_square / (2 * D ** 2))

    return template * fshift

def gaussian_filter_low_f(fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    template = 1 - np.exp(- dis_square / (2 * D ** 2)) # 高斯过滤器

    return template * fshift

def circle_filter_high_f(img, fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift


def circle_filter_low_f(img, fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg

def get_low_high_f(img, radius_ratio, D):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    hight_parts_fshift = circle_filter_low_f(img, fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = circle_filter_high_f(img, fshift.copy(), radius_ratio=radius_ratio)
    hight_parts_fshift =  gaussian_filter_low_f(fshift.copy(), D=D)
    low_parts_fshift = gaussian_filter_high_f(fshift.copy(), D=D)

    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


# 频域中使用高斯滤波器能更好的减少振铃效应
if __name__ == '__main__':
    radius_ratio = 0.5  # 圆形过滤器的半径：ratio * w/2
    D = 5              # 高斯过滤器的截止频率：2 5 10 20 50 ，越小越模糊信息越少
    img = cv2.imread('C:\\Users\\neosunjz\\Desktop\\ican\\6.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    low_freq_part_img, high_freq_part_img = get_low_high_f(img, radius_ratio=radius_ratio, D=D)  # multi channel or single

    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(low_freq_part_img, 'gray'), plt.title('low_freq_img')
    plt.axis('off')
    plt.subplot(133), plt.imshow(high_freq_part_img, 'gray'), plt.title('high_freq_img')
    plt.axis('off')
    plt.show()