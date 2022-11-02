from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args, get_model
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader, get_dataset
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import add_adv_example_file_log, set_adv_example_file_cost_time
from CANARY_SEFI.evaluator.logger.attack_info_handler import add_attack_log
from CANARY_SEFI.evaluator.monitor.attack_effect import time_cost_statistics
from CANARY_SEFI.handler.image_handler.img_io_handler import save_pic_to_temp
from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class AdvAttacker:
    def __init__(self, atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info=None, run_device=None):
        self.atk_component = SEFI_component_manager.attack_method_list.get(atk_name)
        # 攻击处理参数JSON转DICT
        self.atk_args_dict = build_dict_with_json_args(self.atk_component, "attack", atk_args, run_device)
        # 是否不需要传入模型
        no_model = config_manager.config.get("attackConfig", {}).get(atk_name, {}).get("no_model", False)
        if no_model is not True:
            self.atk_args_dict['model'] = get_model(model_name, model_args, run_device)
        model_component = SEFI_component_manager.model_list.get(model_name)

        # 图片处理参数JSON转DICT
        self.img_proc_args_dict = build_dict_with_json_args(model_component, "img_processing", img_proc_args, run_device)
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
            # 攻击类初始化方法
            self.atk_init = self.atk_component.get('attack_init', None)
            # 初始化类
            if self.atk_init is not None and dataset_info is not None:
                self.atk_init(self.attacker_class, self.init_with_dataset(dataset_info, self.img_preprocessor, self.img_proc_args_dict) )

            # 扰动变量名称
            self.perturbation_budget_var_name = self.atk_component.get('attacker_class').get('perturbation_budget_var_name')
        else:
            self.perturbation_budget_var_name = self.atk_component.get('perturbation_budget_var_name')

    @staticmethod
    def init_with_dataset(dataset_info, img_preprocessor, img_proc_args_dict):
        ori_dataset = get_dataset(dataset_info)

        class PreprocessDataset:
            def __init__(self):
                self.ori_dataset = ori_dataset
                self.img_preprocessor = img_preprocessor
                self.img_proc_args_dict = img_proc_args_dict

            def __getitem__(self, index):
                img, label = self.ori_dataset.__getitem__(index)
                if self.img_preprocessor is not None:  # 图片预处理器存在
                    img = self.img_preprocessor(img, self.img_proc_args_dict)
                return img, label

            def __len__(self):
                return ori_dataset.__len__()
        return PreprocessDataset()

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
        check_cuda_memory_alloc_status(empty_cache=True)
        return adv_result

    def destroy(self):
        del self.attacker_class
        check_cuda_memory_alloc_status(empty_cache=True)


def adv_attack_4_img_batch(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info,
                           each_img_finish_callback=None, batch_size=1, completed_num=0, run_device=None):

    adv_img_id_list = []
    # 构建攻击者
    adv_attacker = AdvAttacker(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info, run_device)

    # 写入日志
    atk_perturbation_budget = atk_args[adv_attacker.perturbation_budget_var_name] \
        if adv_attacker.perturbation_budget_var_name is not None else None

    attack_id = add_attack_log(atk_name, model_name, atk_perturbation_budget=atk_perturbation_budget)

    # 攻击单图片迭代函数
    def attack_iterator(imgs, img_log_ids, img_labels, save_raw_data=True):
        # 执行攻击
        adv_results = adv_attacker.adv_attack_4_img(imgs, img_labels)

        # batch分割
        for index in range(len(adv_results)):

            adv_result = adv_results[index]
            img_log_id = img_log_ids[index]

            # 保存至临时文件夹
            # 因为直接转储为PNG会导致精度丢失，产生很多奇怪的结论
            img_file_name = "adv_{}.png".format(img_log_id)
            save_pic_to_temp(str(attack_id) + "/", img_file_name, adv_result)

            raw_file_name = None
            if save_raw_data:
                raw_file_name = "adv_raw_{}.npy".format(img_log_id)
                save_pic_to_temp(str(attack_id) + "/", raw_file_name, adv_result, save_as_numpy_array=True)

            # 写入日志
            adv_img_id = add_adv_example_file_log(attack_id, img_log_id, img_file_name, raw_file_name)
            set_adv_example_file_cost_time(adv_img_id, adv_attacker.cost_time)

            adv_img_id_list.append(adv_img_id)

            if each_img_finish_callback is not None:
                each_img_finish_callback(imgs[index], adv_result)

    dataset_image_reader(attack_iterator, dataset_info, batch_size, completed_num)
    task_manager.sys_log_logger.update_finish_status(True)

    adv_attacker.destroy()
    del adv_attacker
    return adv_img_id_list
