import torch

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args, get_model
from CANARY_SEFI.core.function.dataset_function import dataset_image_reader
from CANARY_SEFI.evaluator.logger.attack_logger import add_adv_build_log, add_attack_log, add_adv_da_log
from CANARY_SEFI.evaluator.monitor.attack_effect import time_cost_statistics
from CANARY_SEFI.evaluator.tester.adv_disturbance_aware import AdvDisturbanceAwareTester
from CANARY_SEFI.handler.image_handler.img_io_handler import save_pic_to_temp


class AdvAttacker:
    def __init__(self, atk_name, atk_args, model_name, model_args, img_proc_args):
        self.atk_component = SEFI_component_manager.attack_method_list.get(atk_name)
        # 攻击处理参数JSON转DICT
        self.atk_args_dict = build_dict_with_json_args(self.atk_component, "attack", atk_args)
        # 是否不需要传入模型（例如当这个模型是黑盒时）
        no_model = config_manager.config.get("attackConfig", {}).get(atk_name, {}).get("no_model", False)
        model = get_model(model_name, model_args)
        if no_model is not True:
            self.atk_args_dict['model'] = model
        model_component = SEFI_component_manager.model_list.get(model_name)
        # 图片处理参数JSON转DICT
        self.img_proc_args_dict = build_dict_with_json_args(model_component, "img_processing", img_proc_args)
        # 图片预处理
        self.img_preprocessor = model_component.get("img_preprocessor")
        # 结果处理
        self.img_reverse_processor = model_component.get("img_reverse_processor")

        self.atk_func = self.atk_component.get('attack_func')
        # 增加计时修饰
        self.cost_time = 0.0
        self.atk_func = time_cost_statistics(self)(self.atk_func)

        # 判断攻击方法的构造模式
        if self.atk_component.get('is_inclass') is True:
            # 构造类传入
            attacker_class_builder = self.atk_component.get('attacker_class').get('class')
            self.attacker_class = attacker_class_builder(**self.atk_args_dict)

    def adv_attack_4_img(self, ori_img, ori_label):
        img = ori_img
        if self.img_preprocessor is not None:  # 图片预处理器存在
            img = self.img_preprocessor(ori_img, self.img_proc_args_dict)
            # 不存在图片预处理器则不对图片进行任何变动直接传入
        # 开始攻击
        # 判断攻击方法的构造模式
        if self.atk_component.get('is_inclass') is True:
            # 构造类传入
            adv_result = self.atk_func(self.attacker_class, img, ori_label)
        else:
            adv_result = self.atk_func(self.atk_args_dict, img, ori_label)

        # 结果处理（一般是图片逆处理器）
        if self.img_reverse_processor is not None:
            adv_result = self.img_reverse_processor(adv_result, self.atk_args_dict)

        # 不存在结果处理器则直接返回
        torch.cuda.empty_cache()
        return adv_result


def adv_attack_4_img_batch(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info,
                           each_img_finish_callback=None):
    adv_img_id_list = []
    adv_attacker = AdvAttacker(atk_name, atk_args, model_name, model_args, img_proc_args)

    def attack_iterator(img, img_log_id, img_label):
        # 执行攻击
        adv_result = adv_attacker.adv_attack_4_img(img, img_label)

        # 保存至临时文件夹
        img_name = "adv_{}_{}.png".format(str(batch_flag.batch_id), str(img_log_id))
        save_pic_to_temp(batch_flag.batch_id, img_name, adv_result)

        # 写入日志
        adv_img_id = add_attack_log(atk_name, model_name, img_log_id, img_name)
        adv_img_id_list.append(adv_img_id)
        add_adv_build_log(adv_img_id, adv_attacker.cost_time)

        if each_img_finish_callback is not None:
            each_img_finish_callback(img, adv_result)

        # 对抗样本综合测试
        adv_da_tester = AdvDisturbanceAwareTester(img_name)
        adv_da_test_result = adv_da_tester.test_all(img, adv_result)

        # 写入日志
        add_adv_da_log(adv_img_id, adv_da_test_result)

    dataset_image_reader(attack_iterator, dataset_info)

    return adv_img_id_list
