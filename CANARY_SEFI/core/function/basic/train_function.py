import copy
import math
import random
import os
import torch
import numpy as np

from CANARY_SEFI.handler.model_weight_handler.weight_file_io_handler import save_weight_to_temp
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args, get_model
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader, get_dataset
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import add_adv_example_file_log, \
    set_adv_example_file_cost_time, set_adv_example_file_query_num
from CANARY_SEFI.evaluator.logger.attack_info_handler import add_attack_log
from CANARY_SEFI.evaluator.monitor.attack_effect import time_cost_statistics
from CANARY_SEFI.handler.image_handler.img_io_handler import save_pic_to_temp
from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class Adversarial_Trainer:
    def __init__(self, defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info=None,
                 each_epoch_finish_callback=None, run_device=None):
        self.defense_component = SEFI_component_manager.defense_method_list.get(defense_name)
        # 攻击处理参数JSON转DICT
        self.defense_args_dict = build_dict_with_json_args(self.defense_component, "defense", defense_args, run_device)

        # TODO 需要修改
        model_component = SEFI_component_manager.model_list.get(model_name)
        self.defense_model = get_model(model_name, model_args, run_device, model_query_logger=None)
        # 图片处理参数JSON转DICT
        self.img_proc_args_dict = build_dict_with_json_args(model_component, "img_processing", img_proc_args,
                                                            run_device)
        # 图片预处理
        self.img_preprocessor = model_component.get("img_preprocessor")

        self.defense_func = self.defense_component.get('defense_func')
        self.each_epoch_finish_callback = each_epoch_finish_callback

        # 判断攻击方法的构造模式
        if self.defense_component.get('is_inclass') is True:
            # 构造类传入
            defense_class_builder = self.defense_component.get('defense_class').get('class')
            self.defense_class = defense_class_builder(**self.defense_args_dict)
            # 防御类初始化方法
            self.defense_init = self.defense_component.get('defense_init', None)

        self.random = random.Random(task_manager.task_token)
        #
        # self.ori_dataset = get_dataset(dataset_info)
        self.dataset_info = dataset_info
        self.img_array = []
        self.label_array = []

    def ori_dataset_preprocess(self):
        for (image, label) in self.ori_dataset:
            img = self.img_preprocessor(np.array(image), self.img_proc_args_dict)
            self.img_array.append(img)
            label = torch.LongTensor([label])
            self.label_array.append(label)

    @staticmethod
    def init_with_dataset(dataset_info, img_preprocessor, img_proc_args_dict):
        ori_dataset = get_dataset(dataset_info)

        class PreprocessDataset:
            def __init__(self):
                self.ori_dataset = ori_dataset
                self.img_preprocessor = img_preprocessor
                self.img_proc_args_dict = img_proc_args_dict
                self.batch_size = self.img_proc_args_dict.get('batch_size', 1)
                self.dataset_size = dataset_info.dataset_size

            def __getitem__(self, index):
                imgs = []
                labels = []
                for num in range(index * self.batch_size,
                                 min((index + 1) * self.batch_size, self.dataset_size)):
                    img, label = self.ori_dataset.__getitem__(num)
                    img = np.array(img)
                    imgs.append(img)
                    labels.append(label)

                if self.img_preprocessor is not None:  # 图片预处理器存在
                    img_batch = self.img_preprocessor(imgs, self.img_proc_args_dict)
                label_batch = torch.LongTensor(labels)
                return img_batch, label_batch

            def __len__(self):
                all_batch = int(math.ceil(self.dataset_size/ self.batch_size))
                return all_batch

        return PreprocessDataset()

    def adv_defense_training_4_img(self):
        # self.ori_dataset_preprocess()
        dataset = self.init_with_dataset(self.dataset_info, self.img_preprocessor, self.img_proc_args_dict)
        if self.defense_component.get('is_inclass') is True:
            weight = self.defense_func(self.defense_class, self.defense_model, dataset,
                                       self.each_epoch_finish_callback)
        else:
            weight = self.defense_func(self.defense_class, self.defense_model, self.img_array, self.label_array,
                                       self.each_epoch_finish_callback)
        return weight

    def destroy(self):
        del self.defense_class
        check_cuda_memory_alloc_status(empty_cache=True)


def adv_defense_4_img_batch(defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info,
                            each_epoch_finish_callback=None, run_device=None):
    # 构建防御训练器
    adv_defense = Adversarial_Trainer(defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info,
                                      each_epoch_finish_callback, run_device)

    weight = adv_defense.adv_defense_training_4_img()
    # Save model
    file_name = "AT_" + model_name + "_" + dataset_info.dataset_name + "_" + task_manager.task_token + ".pt"
    save_weight_to_temp(file_path=model_name + '/', file_name=file_name, weight=weight)
    task_manager.sys_log_logger.update_finish_status(True)
    adv_defense.destroy()
    del adv_defense
    return file_name
