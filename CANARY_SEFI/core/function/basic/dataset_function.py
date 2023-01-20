import math

import cv2
import numpy
import numpy as np

from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import add_img_log, find_img_log_by_id
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_temp


def get_dataset(dataset_info):
    dataset_component = SEFI_component_manager.dataset_list.get(dataset_info.dataset_name)
    if dataset_component is not None:
        dataset_getter = dataset_component.get("dataset_getter_handler")
    else:
        raise Exception("[ Logic Error ] No dataset loader found!")

    dataset_path = config_manager.config.get("dataset", {}).get(dataset_info.dataset_name, {}).get("path", None)
    dataset = dataset_getter(dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size)
    return dataset


# 传入dataset_size，根据dataset_size划分数据集子集，并读入全部的子集（特别的，全部读入则传入数据集原始大小即可）
# 不再提供默认的数据集读入程序
def dataset_image_reader(iterator, dataset_info, batch_size=1, completed_num=0):

    # 对抗样本数据集读入
    if dataset_info.dataset_type != DatasetType.NORMAL:
        adv_dataset_image_reader(iterator, dataset_info, batch_size, completed_num)
        return

    dataset = get_dataset(dataset_info)

    def handler_img(_img_array, _img_label_array, _img_cursor_array):
        img_log_id_array = []
        for index in range(len(_img_array)):
            # 检查尺寸
            _img_array[index] = limit_img_size(_img_array[index])
            # 写入日志
            img_log_id = add_img_log(_img_label_array[index], _img_cursor_array[index])
            img_log_id_array.append(img_log_id)

        # 调用迭代器,传入图片
        iterator(_img_array, img_log_id_array, _img_label_array)
        # 完成数量增加
        task_manager.sys_log_logger.update_completed_num(len(_img_array))

        del _img_array, img_log_id_array, _img_label_array

    # Batch
    all_batch = int(math.ceil(dataset_info.dataset_size/batch_size))
    completed_batch = int(math.ceil(completed_num / batch_size))

    for batch_cursor in range(all_batch):
        if batch_cursor <= completed_batch - 1:
            continue
        img_array = []
        label_array = []
        img_cursor_array = []

        for img_cursor in range(batch_cursor * batch_size, min((batch_cursor+1) * batch_size, dataset_info.dataset_size)):
            img = dataset[int(img_cursor)][0]
            img_label = dataset[int(img_cursor)][1]
            if type(img) != numpy.ndarray:
                img = np.array(dataset[int(img_cursor)][0], dtype=np.uint8)

            img_array.append(img)
            label_array.append(img_label)
            img_cursor_array.append(img_cursor)

        handler_img(img_array, label_array, img_cursor_array)


def dataset_single_image_reader(dataset_info, ori_img_cursor=0):
    dataset = get_dataset(dataset_info)
    img = dataset[int(ori_img_cursor)][0]
    if type(img) != numpy.array:
        img = np.array(img, dtype=np.uint8)
    return limit_img_size(img), dataset[int(ori_img_cursor)][1]


def adv_dataset_image_reader(iterator, dataset_info, batch_size=1, completed_num=0):
    adv_img_type = dataset_info.dataset_type
    adv_img_cursor_list = dataset_info.img_cursor_list

    # Batch
    all_adv_count = dataset_info.dataset_size - completed_num
    for batch_cursor in range(int(math.ceil(all_adv_count/batch_size))):
        adv_img_array = []
        adv_log_id_array = []
        ori_label_array = []

        for adv_cursor in range(batch_cursor * batch_size, min((batch_cursor+1) * batch_size, dataset_info.dataset_size)):
            adv_example_file_log = find_adv_example_file_log_by_id(adv_img_cursor_list[adv_cursor])
            adv_img = adv_dataset_single_image_reader(adv_example_file_log, adv_img_type)
            ori_label = find_img_log_by_id(adv_example_file_log["ori_img_id"])["ori_img_label"]

            if type(adv_img) != numpy.ndarray:
                adv_img = np.array(adv_img, dtype=np.float32)

            adv_img_array.append(adv_img)
            adv_log_id_array.append(adv_img_cursor_list[adv_cursor])
            ori_label_array.append(ori_label)
            del adv_img, ori_label

        iterator(adv_img_array, adv_log_id_array, ori_label_array)
        task_manager.sys_log_logger.update_completed_num(len(adv_img_array))
        del adv_img_array, adv_log_id_array, ori_label_array


def adv_dataset_single_image_reader(adv_file_log, adv_img_type):
    adv_file_path = task_manager.base_temp_path + "pic/" + str(adv_file_log["attack_id"]) + "/"
    if adv_img_type == DatasetType.ADVERSARIAL_EXAMPLE_IMG:
        img = get_pic_nparray_from_temp(adv_file_path, adv_file_log["adv_img_filename"], is_numpy_array_file=False)
    elif adv_img_type == DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA:
        img = get_pic_nparray_from_temp(adv_file_path, adv_file_log["adv_raw_nparray_filename"], is_numpy_array_file=True)
    else:
        raise ValueError("[ Logic Error ] [ READ DATASET IMG ] Wrong dataset type!")
    return img


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