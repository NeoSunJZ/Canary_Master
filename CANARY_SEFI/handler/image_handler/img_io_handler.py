import base64
import os
from io import BytesIO
import cv2
import numpy
import numpy as np
from PIL import Image

from CANARY_SEFI.batch_manager import batch_manager


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
    buff = BytesIO()
    pil_img = Image.fromarray(file_output)
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return "data:image/jpg;base64," + new_image_string


def get_pic_nparray_from_dataset(dataset_path, file_name, is_numpy_array_file=False):
    file_name = img_file_name_handler(file_name, is_numpy_array_file)
    if dataset_path is None:
        raise TypeError("[baispBoot] The dataset path is NOT FOUND, please check your config")
    if not is_numpy_array_file:
        img = cv2.imread(dataset_path + file_name)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return np.loadtxt(dataset_path + file_name)


def save_pic_to_temp(file_name, pic_numpy_array, save_as_numpy_array=False):
    file_name = img_file_name_handler(file_name, save_as_numpy_array)

    if not os.path.exists(batch_manager.base_temp_path + "pic/"):
        os.makedirs(batch_manager.base_temp_path + "pic/")

    full_path = batch_manager.base_temp_path + "pic/" + file_name
    # 判断是否要存储为图片文件(存为图片文件可能存在精度截断)
    if not save_as_numpy_array:
        # 确保存储的是0-255的uint8数据以避免错误
        pic_numpy_array = np.clip(pic_numpy_array, 0, 255).astype(np.uint8)
        cv2.imwrite(full_path, cv2.cvtColor(np.asarray(pic_numpy_array), cv2.COLOR_RGB2BGR))
    else:
        numpy.savetxt(full_path, pic_numpy_array)
    return


def img_file_name_handler(file_name, is_numpy_array_file=False):
    file_name = str(file_name)
    if file_name.find(".") != -1:
        return file_name
    else:
        if is_numpy_array_file:
            return file_name + ".txt"
        else:
            return file_name + ".png"
