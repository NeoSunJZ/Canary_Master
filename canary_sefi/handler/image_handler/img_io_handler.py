import base64
import os
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

from canary_sefi.task_manager import task_manager


def get_nparray_from_file_input(file_input):
    img_pil = Image.open(file_input)
    if len(img_pil.split()) == 4:
        r, g, b, a = img_pil.split()
        img3 = Image.merge("RGB", (r, g, b))
        img = np.asarray(img3)
    else:
        img = np.asarray(img_pil)
    return img


def get_pic_base64_from_nparray(file_output):
    buffer = BytesIO()
    pil_img = Image.fromarray(file_output)
    pil_img.save(buffer, format="JPEG")
    result = pic_buffer_to_base64(buffer)
    del buffer
    return result


def pic_buffer_to_base64(buffer):
    new_image_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    del buffer
    return "data:image/jpg;base64," + new_image_string


def get_pic_nparray_from_temp(file_path, file_name, is_numpy_array_file=False, is_gray=False, force_use_provided_path=False):
    file_name = img_file_name_handler(file_name, is_numpy_array_file)
    if not force_use_provided_path:
        full_path = file_path + ("npy/" if is_numpy_array_file else "img/")
    if full_path is None:
        raise RuntimeError("[ Config Error ] The dataset path is NOT FOUND, please check your config")
    if not is_numpy_array_file:
        img = cv2.imread(full_path + file_name)
        if is_gray:
            np_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            np_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np_img
    else:
        np_img = np.load(full_path + file_name)
        return np_img


def save_pic_to_temp(file_path, file_name, pic_numpy_array, save_as_numpy_array=False):
    file_name = img_file_name_handler(file_name, save_as_numpy_array)
    full_path = task_manager.base_temp_path + "pic/" + file_path + ("npy/" if save_as_numpy_array else "img/")
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    full_path = full_path + file_name
    # 判断是否要存储为图片文件(存为图片文件可能存在精度截断)
    if not save_as_numpy_array:
        # 确保存储的是0-255的uint8数据以避免错误
        pic_numpy_array = np.clip(pic_numpy_array, 0, 255).astype(np.uint8)
        if pic_numpy_array.shape[2] == 3:
            cv2.imwrite(full_path, cv2.cvtColor(np.asarray(pic_numpy_array), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(full_path, np.asarray(pic_numpy_array))
    else:
        np.clip(pic_numpy_array, 0, 255).astype(np.float32)
        np.save(full_path, pic_numpy_array)
    return


def img_file_name_handler(file_name, is_numpy_array_file=False):
    file_name = str(file_name)
    if file_name.find(".") != -1:
        return file_name
    else:
        if is_numpy_array_file:
            return file_name + ".bin"
        else:
            return file_name + ".png"
