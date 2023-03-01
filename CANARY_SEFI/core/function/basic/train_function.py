import copy
import random

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
                 run_device=None):
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
        # 结果处理
        self.img_reverse_processor = model_component.get("img_reverse_processor")

        self.defense_func = self.defense_component.get('defense_func')

        # 判断攻击方法的构造模式
        if self.defense_component.get('is_inclass') is True:
            # 构造类传入
            defense_class_builder = self.defense_component.get('defense_class').get('class')
            self.defense_class = defense_class_builder(**self.defense_args_dict)
            # 攻击类初始化方法
            self.defense_init = self.defense_component.get('defense_init', None)

        self.random = random.Random(task_manager.task_token)

        self.ori_dataset = get_dataset(dataset_info)

    def adv_defense_training_4_img(self):
        if self.defense_component.get('is_inclass') is True:
            weight = self.defense_func(self.defense_class, self.defense_model, self.img_preprocessor,
                                       self.img_reverse_processor, self.img_proc_args_dict, self.ori_dataset)
        else:
            weight = self.defense_func(self.defense_args_dict, self.defense_model, self.img_preprocessor,
                                       self.img_reverse_processor, self.img_proc_args_dict, self.ori_dataset)
        return weight

    def destroy(self):
        del self.defense_class
        check_cuda_memory_alloc_status(empty_cache=True)


def adv_defense_4_img_batch(defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info, run_device=None):
    # 构建防御训练器
    adv_defense = Adversarial_Trainer(defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info,
                                      run_device)
    weight = adv_defense.adv_defense_training_4_img()
    adv_defense.destroy()
    del adv_defense
    return None
