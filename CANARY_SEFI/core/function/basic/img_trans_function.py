import numpy as np
import torch

from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo, DatasetType
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id
from CANARY_SEFI.handler.image_handler.img_io_handler import save_pic_to_temp
from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status
from CANARY_SEFI.evaluator.logger.trans_info_handler import add_trans_log
from CANARY_SEFI.evaluator.logger.trans_file_info_handler import add_adv_trans_img_file_log
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_atk_id_and_ori_img_id


class Image_Transformer:
    def __init__(self, trans_name, trans_args, run_device=None):
        self.trans_name = trans_name
        self.trans_args = trans_args
        self.trans_component = SEFI_component_manager.trans_method_list.get(trans_name)
        # 攻击处理参数JSON转DICT
        self.trans_args_dict = build_dict_with_json_args(self.trans_component, "trans", trans_args, run_device)
        self.trans_func = self.trans_component.get("trans_func")

        # 判断转换方法的构造模式
        if self.trans_component.get('is_inclass') is True:
            # 构造类传入
            trans_class_builder = self.trans_component.get('trans_class').get('class')
            self.trans_class = trans_class_builder(**self.trans_args_dict)
            # 转换类初始化方法
            self.trans_init = self.trans_component.get('trans_init', None)

    def adv_trans_4_img(self, img):
        # TODO 修改
        if self.trans_component.get('is_inclass') is True:
            img_trans = self.trans_func(self.trans_class, img)
        else:
            img_trans = self.trans_func(self.trans_class, img)
        return img_trans

    def destroy(self):
        del self.trans_class
        check_cuda_memory_alloc_status(empty_cache=True)


def adv_trans_4_img_batch(trans_name, trans_args, atk_log, run_device=None):
    trans_img_id_list = []
    # 查找攻击样本日志
    attack_id = atk_log['attack_id']
    all_adv_log = find_adv_example_file_logs_by_attack_id(attack_id)

    adv_img_cursor_list = []
    for adv_log in all_adv_log:
        adv_img_cursor_list.append(adv_log["adv_img_file_id"])

    # 读取攻击样本
    adv_dataset_info = DatasetInfo(None, None, None, adv_img_cursor_list)
    adv_dataset_info.dataset_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA

    # 构建图片转换器
    adv_trans = Image_Transformer(trans_name, trans_args, run_device)

    # 写入日志
    trans_id = add_trans_log(attack_id, trans_name)

    save_path = str(attack_id) + "/trans/" + str(trans_id) + "/"

    def trans_iterator(imgs, img_log_ids, img_labels, save_raw_data=True):
        for index in range(len(imgs)):
            imgs[index] = adv_trans.adv_trans_4_img(imgs[index])
            img_log_id = img_log_ids[index]
            adv_img_file_log = find_adv_example_file_log_by_atk_id_and_ori_img_id(attack_id, img_log_id)
            adv_img_file_id = adv_img_file_log['adv_img_file_id']
            if torch.is_tensor(imgs[index]):
                imgs[index] = imgs[index].numpy()
            trans_file_name = "adv_trans_{}.npy".format(img_log_ids[index])
            save_pic_to_temp(save_path, trans_file_name, imgs[index], save_as_numpy_array=True)

            # 写入日志
            adv_trans_img_file_id = add_adv_trans_img_file_log(trans_id, attack_id, adv_img_file_id, trans_file_name)
            trans_img_id_list.append(adv_trans_img_file_id)

    dataset_image_reader(trans_iterator, adv_dataset_info)

    adv_trans.destroy()
    del adv_trans
    return trans_img_id_list
