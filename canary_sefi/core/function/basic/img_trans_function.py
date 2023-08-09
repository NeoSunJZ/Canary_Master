import copy

import numpy as np
import torch

from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, SubComponentType, \
    TransComponentAttributeType
from canary_sefi.entity.dataset_info_entity import DatasetInfo, DatasetType
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.component.default_component.params_handler import build_dict_with_json_args
from canary_sefi.core.function.basic.dataset.dataset_function import dataset_image_reader
from canary_sefi.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id, \
    find_adv_example_file_log_by_id
from canary_sefi.handler.image_handler.img_io_handler import save_pic_to_temp
from canary_sefi.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status
from canary_sefi.evaluator.logger.trans_file_info_handler import add_adv_trans_img_file_log


class Image_Transformer:
    def __init__(self, trans_name, trans_args, run_device=None):
        self.trans_name = trans_name
        self.trans_args = trans_args
        self.trans_component = SEFI_component_manager.trans_method_list.get(trans_name)
        # 攻击处理参数JSON转DICT
        self.trans_args_dict = build_dict_with_json_args(self.trans_component,
                                                         ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                                         trans_args, run_device)
        self.trans_func = self.trans_component.get(SubComponentType.TRANS_FUNC)

        # 判断转换方法的构造模式
        if self.trans_component.get(TransComponentAttributeType.IS_INCLASS) is True:
            # 构造类传入
            trans_class_builder = self.trans_component.get(SubComponentType.TRANS_CLASS)
            self.trans_class = trans_class_builder(**self.trans_args_dict)
            # 转换类初始化方法
            self.trans_init = self.trans_component.get(SubComponentType.TRANS_INIT, None)

    def adv_trans_4_img(self, img):
        if self.trans_component.get(TransComponentAttributeType.IS_INCLASS) is True:
            img_trans = self.trans_func(self.trans_class, img)
        else:
            img_trans = self.trans_func(self.trans_args_dict, img)
        return img_trans

    def destroy(self):
        del self.trans_class
        check_cuda_memory_alloc_status(empty_cache=True)


def adv_trans_4_img_batch(trans_name, trans_args, atk_log, run_device=None, use_raw_nparray_data=False):
    trans_img_id_list = []
    # 查找攻击样本日志
    attack_id = atk_log['attack_id']
    all_adv_log = find_adv_example_file_logs_by_attack_id(attack_id)

    adv_img_cursor_list = []
    for adv_log in all_adv_log:
        adv_img_cursor_list.append(adv_log["adv_img_file_id"])

    # 读取攻击样本
    adv_dataset_info = DatasetInfo(None, None,
                                   dataset_type=DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA
                                   if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG,
                                   img_cursor_list=adv_img_cursor_list)

    # 构建图片转换器
    adv_trans = Image_Transformer(trans_name, trans_args, run_device)

    save_path = str(attack_id) + "/trans/" + trans_name + "/"

    def trans_iterator(imgs, img_log_ids, img_labels, save_raw_data=True):
        # 生成防御样本
        trans_results = adv_trans.adv_trans_4_img(copy.deepcopy(imgs))
        for index in range(len(trans_results)):
            img_log_id = img_log_ids[index]
            adv_img_file_log = find_adv_example_file_log_by_id(img_log_id)
            adv_img_file_id = adv_img_file_log['adv_img_file_id']
            if torch.is_tensor(trans_results[index]):
                trans_results[index] = trans_results[index].numpy()
            trans_img_file_name = "adv_trans_{}.png".format(img_log_id)
            save_pic_to_temp(save_path, trans_img_file_name, trans_results[index], save_as_numpy_array=False)
            raw_file_name = None
            if save_raw_data:
                raw_file_name = "adv_trans_{}.npy".format(img_log_id)
                save_pic_to_temp(save_path, raw_file_name, trans_results[index], save_as_numpy_array=True)

            # 写入日志
            adv_trans_img_file_id = add_adv_trans_img_file_log(trans_name, attack_id, adv_img_file_id,
                                                               trans_img_file_name, raw_file_name)
            trans_img_id_list.append(adv_trans_img_file_id)

    dataset_image_reader(trans_iterator, adv_dataset_info)

    adv_trans.destroy()
    del adv_trans
    return trans_img_id_list
