import math

import numpy as np

from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.basic.dataset.memory_cache import memory_cache
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.evaluator.logger.trans_file_info_handler import find_adv_trans_file_log_by_id
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_temp
from CANARY_SEFI.task_manager import task_manager


def trans_dataset_image_reader(iterator, dataset_info, batch_size=1, completed_num=0, disable_memory_cache=False):
    trans_img_type = dataset_info.dataset_type
    trans_img_cursor_list = dataset_info.img_cursor_list

    # Batch
    all_trans_count = dataset_info.dataset_size - completed_num
    for batch_cursor in range(int(math.ceil(all_trans_count / batch_size))):
        trans_img_array = []
        trans_log_id_array = []
        ori_label_array = []

        for trans_cursor in range(batch_cursor * batch_size,
                                  min((batch_cursor + 1) * batch_size, dataset_info.dataset_size)):
            trans_img, ori_label = get_trans_img(trans_img_cursor_list[trans_cursor], trans_img_type, disable_memory_cache)
            if type(trans_img) != np.ndarray:
                trans_img = np.array(trans_img, dtype=np.float32)

            trans_img_array.append(trans_img)
            trans_log_id_array.append(trans_img_cursor_list[trans_cursor])
            ori_label_array.append(ori_label)
            del trans_img, ori_label

        iterator(trans_img_array, trans_log_id_array, ori_label_array)
        task_manager.sys_log_logger.update_completed_num(len(trans_img_array))
        del trans_img_array, trans_log_id_array, ori_label_array


def trans_dataset_single_image_reader(trans_file_log, trans_img_type):
    trans_file_path = task_manager.base_temp_path + "pic/" + str(trans_file_log["attack_id"]) + "/trans/" + str(
        trans_file_log["trans_name"]) + "/"
    if trans_img_type == DatasetType.TRANSFORM_IMG:
        img = get_pic_nparray_from_temp(trans_file_path, trans_file_log["adv_trans_img_filename"],
                                        is_numpy_array_file=False)
    elif trans_img_type == DatasetType.TRANSFORM_RAW_DATA:
        img = get_pic_nparray_from_temp(trans_file_path, trans_file_log["adv_trans_img_filename"],
                                        is_numpy_array_file=True)
    else:
        raise ValueError("[ Logic Error ] [ READ DATASET IMG ] Wrong dataset type!")
    return img


def get_trans_img(trans_img_id, trans_img_type, disable_memory_cache=False):
    # 若禁用内存缓存增强
    if not config_manager.config.get("system", {}).get("use_file_memory_cache", False) or disable_memory_cache:
        trans_file_log = find_adv_trans_file_log_by_id(trans_img_id)
        adv_example_file_log = find_adv_example_file_log_by_id(trans_file_log['adv_img_file_id'])
        ori_label = find_img_log_by_id(adv_example_file_log["ori_img_id"])["ori_img_label"]
        return trans_dataset_single_image_reader(trans_file_log, trans_img_type), ori_label

    trans_img_data = memory_cache.trans_img_list.get(trans_img_id, None)
    if trans_img_data is None:
        trans_file_log = find_adv_trans_file_log_by_id(trans_img_id)
        trans_img = trans_dataset_single_image_reader(trans_file_log, trans_img_type)
        adv_example_file_log = find_adv_example_file_log_by_id(trans_file_log['adv_img_file_id'])
        ori_label = find_img_log_by_id(adv_example_file_log["ori_img_id"])["ori_img_label"]

        # 存入临时缓存
        memory_cache.trans_img_list[trans_img_id] = {
            "trans_img": trans_img,
            "ori_label": ori_label,
        }
    else:
        trans_img = trans_img_data.get("trans_img")
        ori_label = trans_img_data.get("ori_label")
    return trans_img, ori_label
