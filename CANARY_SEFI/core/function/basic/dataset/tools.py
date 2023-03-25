import cv2

from CANARY_SEFI.core.config.config_manager import config_manager


def limit_img_size(img):
    # 检查尺寸
    limited_img_size = config_manager.config.get("system", {}).get("limited_read_img_size", None)
    if limited_img_size is not None:
        height, width = img.shape[0], img.shape[1]
        if height >= limited_img_size and height >= width:  # 高度超限
            scale = height / limited_img_size
            width_size = int(width / scale)
            # 缩放之
            img = cv2.resize(img, (width_size, limited_img_size), interpolation=cv2.INTER_CUBIC)
        elif width >= limited_img_size and width >= height:  # 宽度超限
            scale = width / limited_img_size
            height_size = int(height / scale)
            # 缩放之
            img = cv2.resize(img, (limited_img_size, height_size), interpolation=cv2.INTER_CUBIC)
    return img