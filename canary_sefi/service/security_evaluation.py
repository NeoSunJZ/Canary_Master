from canary_sefi.core.function.enum.test_level_enum import TestLevel
from canary_sefi.entity.dataset_info_entity import DatasetType
from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.enum.transfer_attack_type_enum import TransferAttackType
from canary_sefi.core.function.init_dataset import init_dataset
from canary_sefi.core.function.test_and_evaluation import adv_example_generate, model_inference_capability_test, \
    model_inference_capability_evaluation, attack_deflection_capability_test, attack_deflection_capability_evaluation, \
    model_security_synthetical_capability_evaluation, \
    adv_example_generate_with_perturbation_increment, attack_deflection_capability_test_with_perturbation_increment, \
    attack_adv_example_da_test_with_perturbation_increment, attack_capability_evaluation_with_perturbation_increment, \
    attack_adv_example_da_and_cost_evaluation, attack_adv_example_comparative_test, \
    attack_synthetical_capability_evaluation, \
    trans_deflection_capability_test, trans_deflection_capability_evaluation, adv_trans_generate, adv_defense_training
from canary_sefi.handler.json_handler.json_io_handler import save_info_to_json_file, get_info_from_json_file


class SecurityEvaluation:

    def __init__(self, config=None, config_handler=None):
        if config is None:
            config = get_info_from_json_file(task_manager.base_temp_path, "config.json")
        else:
            save_info_to_json_file(config, task_manager.base_temp_path, "config.json")
        if config_handler is not None:
            config = config_handler(config)
            save_info_to_json_file(config, task_manager.base_temp_path, "config.json")

        # 初始化数据集
        if isinstance(config.get("dataset"), str):
            self.dataset_info = init_dataset(config.get("dataset"), config.get("dataset_size", None),
                                             config.get("dataset_seed", None))
        elif isinstance(config.get("dataset"), dict):
            dataset_extra_config = config.get("dataset")
            self.dataset_info = init_dataset(
                dataset_name=dataset_extra_config.get("dataset_name"),
                dataset_size=config.get("dataset_size", None),
                dataset_seed=config.get("dataset_seed", None),
                dataset_path=dataset_extra_config.get("dataset_path", None),
                dataset_folder=dataset_extra_config.get("dataset_folder", None),
                dataset_type=dataset_extra_config.get("dataset_type", DatasetType.NORMAL),
                n_classes=dataset_extra_config.get("n_classes", None),
                is_gray=dataset_extra_config.get("is_gray", False),
                is_fast_test=dataset_extra_config.get("is_fast_test", False)
            )

        self.model_list = config.get("model_list", None)
        self.attacker_list = config.get("attacker_list", None)
        self.trans_list = config.get("trans_list", None)
        self.defense_AT_list = config.get("defense_AT_list", None)

        self.transfer_attack_test_mode = TransferAttackType(config.get("transfer_attack_test_mode", "NOT"))
        self.transfer_attack_test_on_model_list = config.get("transfer_attack_test_on_model_list", {})

        self.model_config = config.get("model_config", None)
        self.attacker_config = config.get("attacker_config", None)
        self.trans_config = config.get("trans_config", None)
        self.defense_AT_config = config.get("defense_AT_config", None)
        self.img_proc_config = config.get("img_proc_config", None)
        self.img_train_proc_config = config.get("img_train_proc_config", None)

        self.perturbation_increment_config = config.get("perturbation_increment_config", None)

        self.inference_batch_config = config.get("inference_batch_config", {})
        self.adv_example_generate_batch_config = config.get("adv_example_generate_batch_config", {})

    def adv_example_generate(self):
        # 生成对抗样本与对抗样本质量分析
        adv_example_generate(self.dataset_info, self.attacker_list, self.attacker_config, self.model_config,
                             self.img_proc_config, self.adv_example_generate_batch_config)
        task_manager.test_data_logger.finish()

    def model_inference_capability_test_and_evaluation(self):
        # 模型推理能力测试
        model_inference_capability_test(self.dataset_info, self.model_list, self.model_config, self.img_proc_config,
                                        self.inference_batch_config)
        # 模型推理能力评估
        model_inference_capability_evaluation(self.model_list)
        task_manager.test_data_logger.finish()

    def attack_test_and_evaluation(self, use_raw_nparray_data=False, transfer_test_level=TestLevel.ESSENTIAL_ONLY):
        # 攻击偏转能力测试
        attack_deflection_capability_test(self.attacker_list, self.dataset_info, self.model_config,
                                          self.img_proc_config,
                                          self.inference_batch_config,
                                          self.transfer_attack_test_mode, self.transfer_attack_test_on_model_list,
                                          use_raw_nparray_data, transfer_test_level)
        # 攻击方法推理偏转效果/模型注意力偏转效果评估
        attack_deflection_capability_evaluation(self.attacker_list, self.dataset_info, use_raw_nparray_data)
        # 攻击方法生成对抗样本综合对比测试(图像相似性/模型注意力差异对比/像素差异对比)
        attack_adv_example_comparative_test(self.attacker_list, self.dataset_info, use_raw_nparray_data)
        # 攻击方法生成对抗样本图像相似性(扰动距离)/生成代价评估
        attack_adv_example_da_and_cost_evaluation(self.attacker_list, use_raw_nparray_data)
        task_manager.test_data_logger.finish()

    def capability_evaluation(self, use_raw_nparray_data=False):
        attack_synthetical_capability_evaluation(self.attacker_list, use_raw_nparray_data)
        model_security_synthetical_capability_evaluation(self.model_list, use_raw_nparray_data)

    def attack_full_test(self, use_img_file=True, use_raw_nparray_data=False,
                         transfer_test_level=TestLevel.ESSENTIAL_ONLY):
        skip_step_list, finish_callback = task_manager.multi_database.get_skip_step()
        if not skip_step_list["model_inference_capability_test_and_evaluation"]:
            self.model_inference_capability_test_and_evaluation()
        if not skip_step_list["adv_example_generate"]:
            self.adv_example_generate()

        if not use_img_file and not use_raw_nparray_data:
            raise RuntimeError("[ Logic Error ] [ INIT TEST ] At least one format of data should be selected!")
        if use_img_file:
            if not skip_step_list["attack_test_and_evaluation"]:
                self.attack_test_and_evaluation(use_raw_nparray_data=False, transfer_test_level=transfer_test_level)
            if not skip_step_list["synthetical_capability_evaluation"]:
                # 综合能力测试结果分析
                self.capability_evaluation(use_raw_nparray_data=False)
        if use_raw_nparray_data:
            if not skip_step_list["attack_test_and_evaluation"]:
                self.attack_test_and_evaluation(use_raw_nparray_data=True, transfer_test_level=transfer_test_level)
            if not skip_step_list["synthetical_capability_evaluation"]:
                # 综合能力测试结果分析
                self.capability_evaluation(use_raw_nparray_data=True)
        # 流程结束回调
        finish_callback()

    def attack_perturbation_increment_test(self, use_img_file=True, use_raw_nparray_data=False):
        # 递增扰动生成对抗样本
        adv_example_generate_with_perturbation_increment(self.dataset_info, self.attacker_list, self.attacker_config,
                                                         self.model_config, self.img_proc_config,
                                                         self.adv_example_generate_batch_config,
                                                         self.perturbation_increment_config)
        # 模型推理能力测试
        model_inference_capability_test(self.dataset_info, self.model_list, self.model_config, self.img_proc_config,
                                        self.inference_batch_config)

        if use_img_file:
            # 测试
            attack_deflection_capability_test_with_perturbation_increment(self.dataset_info, self.attacker_list, self.model_config,
                                                                          self.img_proc_config,
                                                                          use_raw_nparray_data=False)
            attack_adv_example_da_test_with_perturbation_increment(self.attacker_list, self.dataset_info,
                                                                   use_raw_nparray_data=False)
            # 结果分析
            attack_capability_evaluation_with_perturbation_increment(self.attacker_list, self.dataset_info,
                                                                     use_raw_nparray_data=False)
        if use_raw_nparray_data:
            # 测试
            attack_deflection_capability_test_with_perturbation_increment(self.dataset_info, self.attacker_list, self.model_config,
                                                                          self.img_proc_config,
                                                                          use_raw_nparray_data=True)
            attack_adv_example_da_test_with_perturbation_increment(self.attacker_list, self.dataset_info,
                                                                   use_raw_nparray_data=True)
            # 结果分析
            attack_capability_evaluation_with_perturbation_increment(self.attacker_list, self.dataset_info,
                                                                     use_raw_nparray_data=True)

        task_manager.test_data_logger.finish()

    def adv_trans_generate(self, use_raw_nparray_data=False):
        # 生成图片预处理防御样本
        adv_trans_generate(self.trans_list, self.trans_config, use_raw_nparray_data)
        task_manager.test_data_logger.finish()

    def trans_test_and_evaluation(self, use_raw_nparray_data=False, transfer_test_level=TestLevel.ESSENTIAL_ONLY):
        # 防御样本攻击偏转能力测试
        trans_deflection_capability_test(self.dataset_info, self.trans_list, self.model_config, self.img_proc_config,
                                         self.inference_batch_config,
                                         self.transfer_attack_test_mode, self.transfer_attack_test_on_model_list,
                                         use_raw_nparray_data, transfer_test_level)
        # 防御样本攻击方法推理偏转效果/模型注意力偏转效果评估
        trans_deflection_capability_evaluation(self.trans_list, self.dataset_info, use_raw_nparray_data)
        task_manager.test_data_logger.finish()

    def trans_full_test(self, use_raw_nparray_data=False, transfer_test_level=TestLevel.ESSENTIAL_ONLY):
        self.adv_trans_generate(use_raw_nparray_data)
        self.trans_test_and_evaluation(use_raw_nparray_data, transfer_test_level)

    def adv_defense_training(self):
        # 对抗防御训练
        adv_defense_training(self.dataset_info, self.defense_AT_list, self.defense_AT_config, self.model_config,
                             self.img_train_proc_config)
        task_manager.test_data_logger.finish()

    def defense_test_and_evaluation(self, use_raw_nparray_data=False, transfer_test_level=TestLevel.FULL):
        # 干净图像预测及评估
        model_inference_capability_test(self.dataset_info, self.model_list, self.model_config, self.img_proc_config,
                                        self.inference_batch_config)
        model_inference_capability_evaluation(self.model_list)
        # 对抗样本预测及评估
        adv_example_generate(self.dataset_info, self.attacker_list, self.attacker_config, self.model_config,
                             self.img_proc_config, self.adv_example_generate_batch_config)
        attack_deflection_capability_test(self.attacker_list, self.model_config, self.img_proc_config,
                                          self.inference_batch_config,
                                          self.transfer_attack_test_mode,
                                          self.transfer_attack_test_on_model_list,
                                          use_raw_nparray_data,
                                          transfer_test_level)
        attack_deflection_capability_evaluation(self.attacker_list, self.dataset_info, use_raw_nparray_data)
        task_manager.test_data_logger.finish()

# sys.excepthook = excepthook
