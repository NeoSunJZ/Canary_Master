import math

import numpy as np

from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.basic.dataset.memory_cache import memory_cache
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_temp
from CANARY_SEFI.task_manager import task_manager


def adv_dataset_image_reader(iterator, dataset_info, batch_size=1, completed_num=0, disable_memory_cache=False):
    adv_img_type = dataset_info.dataset_type
    adv_img_cursor_list = dataset_info.img_cursor_list

    # Batch
    all_adv_count = dataset_info.dataset_size - completed_num
    for batch_cursor in range(int(math.ceil(all_adv_count/batch_size))):
        adv_img_array = []
        adv_log_id_array = []
        ori_label_array = []

        for adv_cursor in range(batch_cursor * batch_size, min((batch_cursor+1) * batch_size, dataset_info.dataset_size)):
            adv_img, ori_label = get_adv_img(adv_img_cursor_list[adv_cursor], adv_img_type, disable_memory_cache)
            if type(adv_img) != np.ndarray:
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

def get_adv_img(adv_img_id, adv_img_type, disable_memory_cache=False):
    # 若禁用内存缓存增强
    if not config_manager.config.get("system", {}).get("use_file_memory_cache", False) or disable_memory_cache:
        adv_example_file_log = find_adv_example_file_log_by_id(adv_img_id)
        ori_label = find_img_log_by_id(adv_example_file_log["ori_img_id"])["ori_img_label"]
        return adv_dataset_single_image_reader(adv_example_file_log, adv_img_type), ori_label

    adv_img_data = memory_cache.adv_img_list.get(adv_img_id, None)
    if adv_img_data is None:
        adv_example_file_log = find_adv_example_file_log_by_id(adv_img_id)
        adv_img = adv_dataset_single_image_reader(adv_example_file_log, adv_img_type)
        ori_label = find_img_log_by_id(adv_example_file_log["ori_img_id"])["ori_img_label"]

        # 存入临时缓存
        memory_cache.adv_img_list[adv_img_id] = {
            "adv_img": adv_img,
            "ori_label": ori_label,
        }
    else:
        adv_img = adv_img_data.get("adv_img")
        ori_label = adv_img_data.get("ori_label")
    return adv_img, ori_label