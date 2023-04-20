import math
import random
import torch
import numpy as np

from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, SubComponentType, \
    DefenseComponentAttributeType
from canary_sefi.core.component.default_component.dataset_getter import get_dataset
from canary_sefi.core.component.default_component.model_getter import get_model
from canary_sefi.core.component.default_component.params_handler import build_dict_with_json_args
from canary_sefi.handler.model_weight_handler.weight_file_io_handler import save_weight_to_temp
from canary_sefi.task_manager import task_manager
from canary_sefi.core.component.component_manager import SEFI_component_manager

from canary_sefi.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class Adversarial_Trainer:
    def __init__(self, defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info=None,
                 each_epoch_finish_callback=None, run_device=None):
        self.defense_component = SEFI_component_manager.defense_method_list.get(defense_name)
        # 攻击处理参数JSON转DICT
        self.defense_args_dict = build_dict_with_json_args(self.defense_component,
                                                           ComponentConfigHandlerType.DEFENSE_CONFIG_PARAMS,
                                                           defense_args, run_device)
        model_component = SEFI_component_manager.model_list.get(model_name)
        self.defense_model = get_model(model_name, model_args, run_device)
        # 图片处理参数JSON转DICT
        self.img_proc_args_dict = build_dict_with_json_args(model_component,
                                                            ComponentConfigHandlerType.IMG_PROCESS_CONFIG_PARAMS,
                                                            img_proc_args, run_device)
        # 图片预处理
        self.img_preprocessor = model_component.get(SubComponentType.IMG_PREPROCESSOR)

        self.defense_func = self.defense_component.get(SubComponentType.DEFENSE_FUNC)
        self.each_epoch_finish_callback = each_epoch_finish_callback

        # 判断攻击方法的构造模式
        if self.defense_component.get(DefenseComponentAttributeType.IS_INCLASS) is True:
            # 构造类传入
            defense_class_builder = self.defense_component.get(SubComponentType.DEFENSE_CLASS)
            self.defense_class = defense_class_builder(**self.defense_args_dict)
            # 防御类初始化方法
            self.defense_init = self.defense_component.get(SubComponentType.DEFENSE_INIT, None)

        self.random = random.Random(task_manager.task_token)
        self.dataset_info = dataset_info

    @staticmethod
    def init_with_dataset(ori_dataset, img_preprocessor, img_proc_args_dict):

        class PreprocessDataset:
            def __init__(self):
                self.ori_dataset = ori_dataset
                self.img_preprocessor = img_preprocessor
                self.img_proc_args_dict = img_proc_args_dict
                self.batch_size = self.img_proc_args_dict.get('batch_size', 1)
                self.dataset_size = len(ori_dataset)

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
                all_batch = int(math.ceil(self.dataset_size / self.batch_size))
                return all_batch

        return PreprocessDataset()

    def adv_defense_training_4_img(self):
        train_dataset, val_dataset = get_dataset(self.dataset_info)
        train_dataset = self.init_with_dataset(train_dataset, self.img_preprocessor, self.img_proc_args_dict)
        val_dataset = self.init_with_dataset(val_dataset, self.img_preprocessor, self.img_proc_args_dict)
        if self.defense_component.get(DefenseComponentAttributeType.IS_INCLASS) is True:
            weight = self.defense_func(self.defense_class, self.defense_model, train_dataset, val_dataset,
                                       self.each_epoch_finish_callback)
        else:
            weight = self.defense_func(self.defense_class, self.defense_model, train_dataset, val_dataset,
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
    file_path = model_name + '/' + defense_name + "/"
    file_name = "AT_" + defense_name + '_' + model_name + "_final" + ".pt"
    save_weight_to_temp(model_name=model_name, defense_name=defense_name, epoch_cursor="final", file_path=file_path,
                        file_name=file_name, weight=weight)
    task_manager.sys_log_logger.update_finish_status(True)
    adv_defense.destroy()
    del adv_defense
    return file_name

