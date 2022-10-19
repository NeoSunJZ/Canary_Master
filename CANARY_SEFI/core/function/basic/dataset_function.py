import cv2
import numpy
import numpy as np
from torch.utils.data import DataLoader

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import add_img_log
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_dataset


def get_dataset(dataset_info):
    dataset_component = SEFI_component_manager.dataset_list.get(dataset_info.dataset_name)
    if dataset_component is not None:
        dataset_getter = dataset_component.get("dataset_getter_handler")
    else:
        raise Exception("[ ERROR ] No dataset loader found!")

    dataset_path = config_manager.config.get("dataset", {}).get(dataset_info.dataset_name, {}).get("path", None)
    dataset = dataset_getter(dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size)
    return dataset


# 传入dataset_size，根据dataset_size划分数据集子集，并读入全部的子集（特别的，全部读入则传入数据集原始大小即可）
# 不再提供默认的数据集读入程序
def dataset_image_reader(iterator, dataset_info, completed_num=0):

    # 对抗样本数据集读入
    if dataset_info.dataset_type != DatasetType.NORMAL:
        adv_dataset_image_reader(iterator, dataset_info)
        return

    dataset = get_dataset(dataset_info)

    def handler_img(_img, _img_label, _img_cursor):
        # 检查尺寸
        _img = limit_img_size(_img)
        # 写入日志
        img_log_id = add_img_log(_img_label, _img_cursor)
        # 调用迭代器,传入图片
        iterator(_img, img_log_id, _img_label)
        # 完成数量增加
        batch_manager.sys_log_logger.update_completed_num(1)

    for img_cursor in range(completed_num, dataset_info.dataset_size):
        img = dataset[int(img_cursor)][0]
        if type(img) != numpy.ndarray:
            img = np.array(dataset[int(img_cursor)][0], dtype=np.uint8)

        img_label = dataset[int(img_cursor)][1]
        handler_img(img, img_label, img_cursor)


def dataset_single_image_reader(dataset_info, ori_img_cursor=0):
    dataset = get_dataset(dataset_info)
    img = dataset[int(ori_img_cursor)][0]
    print(type(img))
    if type(img) != numpy.array:
        img = np.array(img, dtype=np.uint8)
    return limit_img_size(img), dataset[int(ori_img_cursor)][1]


def adv_dataset_image_reader(iterator, dataset_info):
    adv_img_type = dataset_info.dataset_type
    adv_img_cursor_list = dataset_info.img_cursor_list

    for i in range(len(adv_img_cursor_list)):
        adv_file_log = find_adv_example_file_log_by_id(adv_img_cursor_list[i])

        img = adv_dataset_single_image_reader(adv_file_log, adv_img_type)
        iterator(img, adv_img_cursor_list[i], None)


        batch_manager.sys_log_logger.update_completed_num(1)


def adv_dataset_single_image_reader(adv_file_log, adv_img_type):
    adv_file_path = batch_manager.base_temp_path + "pic/"
    if adv_img_type == DatasetType.ADVERSARIAL_EXAMPLE_IMG:
        img = get_pic_nparray_from_dataset(adv_file_path, adv_file_log["adv_img_filename"], is_numpy_array_file=False)
    elif adv_img_type == DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA:
        img = get_pic_nparray_from_dataset(adv_file_path, adv_file_log["adv_raw_nparray_filename"],
                                           is_numpy_array_file=True)
    else:
        raise RuntimeError("[ Logic Error ] [ READ DATASET IMG ] Wrong dataset type!")
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