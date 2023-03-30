import math

import numpy
import numpy as np

from CANARY_SEFI.core.component.default_component.dataset_getter import get_dataset
from CANARY_SEFI.core.function.basic.dataset.adv_dataset_function import adv_dataset_image_reader
from CANARY_SEFI.core.function.basic.dataset.memory_cache import memory_cache
from CANARY_SEFI.core.function.basic.dataset.tools import limit_img_size
from CANARY_SEFI.core.function.basic.dataset.trans_dataset_function import trans_dataset_image_reader
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.img_file_info_handler import add_img_log, find_img_log_by_id


# 传入dataset_size，根据dataset_size划分数据集子集，并读入全部的子集（特别的，全部读入则传入数据集原始大小即可）
# 不再提供默认的数据集读入程序
def dataset_image_reader(iterator, dataset_info, batch_size=1, completed_num=0):
    # 对抗样本数据集读入
    if dataset_info.dataset_type == DatasetType.ADVERSARIAL_EXAMPLE_IMG or dataset_info.dataset_type == DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA:
        adv_dataset_image_reader(iterator, dataset_info, batch_size, completed_num)
        return
    if dataset_info.dataset_type == DatasetType.TRANSFORM_IMG or dataset_info.dataset_type == DatasetType.TRANSFORM_RAW_DATA:
        trans_dataset_image_reader(iterator, dataset_info, batch_size, completed_num)
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
    all_batch = int(math.ceil(dataset_info.dataset_size / batch_size))
    completed_batch = int(math.ceil(completed_num / batch_size))

    for batch_cursor in range(all_batch):
        if batch_cursor <= completed_batch - 1:
            continue
        img_array = []
        label_array = []
        img_cursor_array = []

        for img_cursor in range(batch_cursor * batch_size,
                                min((batch_cursor + 1) * batch_size, dataset_info.dataset_size)):
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


def get_ori_img(dataset_info, ori_img_id, disable_memory_cache=False):
    # 若禁用内存缓存增强
    if not config_manager.config.get("system", {}).get("use_file_memory_cache", False) or disable_memory_cache:
        ori_img_log = find_img_log_by_id(ori_img_id)
        return dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])

    ori_img_data = memory_cache.ori_img_list.get(ori_img_id, None)
    if ori_img_data is None:
        ori_img_log = find_img_log_by_id(ori_img_id)
        ori_img, ori_label = dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])
        # 存入临时缓存
        memory_cache.ori_img_list[ori_img_id] = {
            "ori_img": ori_img,
            "ori_label": ori_label,
        }
    else:
        ori_img = ori_img_data.get("ori_img")
        ori_label = ori_img_data.get("ori_label")
    return ori_img, ori_label
