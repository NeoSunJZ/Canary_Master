import base64
import os
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

from CANARY_SEFI.core.config.config_manager import config_manager


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


def get_pic_nparray_from_dataset(dataset_path, pic_name):
    pic_name = img_file_name_handler(pic_name)
    if dataset_path is None:
        raise TypeError("[baispBoot] The dataset path is NOT FOUND, please check your config")
    img = cv2.imread(dataset_path + pic_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_pic_nparray_from_temp(temp_token, pic_name):
    temp_path = config_manager.config.get("temp", "Dataset_Temp/")
    img = cv2.imread(temp_path + temp_token + "/" + pic_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_pic_to_temp(temp_token, pic_name, result):
    pic_name = img_file_name_handler(pic_name)
    temp_path = config_manager.config.get("temp", "Dataset_Temp/")
    if not os.path.exists(temp_path + temp_token):
        os.makedirs(temp_path + temp_token)
    full_path = temp_path + temp_token + "/" + pic_name
    cv2.imwrite(full_path, cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR))
    return


def get_temp_download_url(temp_token):
    return temp_token


def img_file_name_handler(pic_name):
    pic_name = str(pic_name)
    if pic_name.find(".") != -1:
        return pic_name
    else:
        return pic_name + ".png"