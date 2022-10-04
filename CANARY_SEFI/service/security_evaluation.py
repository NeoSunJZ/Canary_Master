import sys
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.enum.transfer_attack_type_enum import TransferAttackType
from CANARY_SEFI.core.function.helper.excepthook import excepthook
from CANARY_SEFI.core.function.init_dataset import init_dataset
from CANARY_SEFI.core.function.test_and_evaluation import explore_attack_perturbation, \
    explore_perturbation_attack_capability_evaluation, adv_example_generate, model_inference_capability_test, \
    model_inference_capability_evaluation, attack_deflection_capability_test, attack_deflection_capability_evaluation, \
    attack_adv_example_da_test, attack_adv_example_da_evaluation, model_security_synthetical_capability_evaluation, \
    explore_perturbation_attack_deflection_capability_test, explore_perturbation_attack_adv_example_da_test
from CANARY_SEFI.handler.json_handler.json_io_handler import save_info_to_json_file, get_info_from_json_file

logger = batch_manager.test_data_logger


class SecurityEvaluation:

    def __init__(self, config=None):
        if config is None:
            config = get_info_from_json_file("config.json")
        else:
            save_info_to_json_file(config, "config.json")
        self.dataset_info = init_dataset(config.get("dataset"), config.get("dataset_size"), config.get("dataset_seed", None))

        self.model_list = config.get("model_list", None)
        self.attacker_list = config.get("attacker_list", None)

        self.transfer_attack_test_mode = TransferAttackType(config.get("transfer_attack_test_mode", "NOT"))
        self.transfer_attack_test_on_model_list = config.get("transfer_attack_test_on_model_list", {})

        self.model_config = config.get("model_config", None)
        self.attacker_config = config.get("attacker_config", None)
        self.img_proc_config = config.get("img_proc_config", None)

        self.explore_perturbation_config = config.get("explore_perturbation_config", None)

    def only_build_adv(self):
        # 生成对抗样本与对抗样本质量分析
        adv_example_generate(self.dataset_info, self.attacker_list, self.attacker_config, self.model_config, self.img_proc_config)
        logger.finish()

    def model_inference_capability_test_and_evaluation(self):
        # 模型推理能力测试
        model_inference_capability_test(self.dataset_info, self.model_list, self.model_config, self.img_proc_config)
        # 模型推理能力评估
        model_inference_capability_evaluation(self.model_list)
        logger.finish()

    def attack_deflection_capability_test_and_evaluation(self, use_raw_nparray_data=False):
        # 攻击偏转能力测试
        attack_deflection_capability_test(self.attacker_list, self.model_config, self.img_proc_config,
                                          self.transfer_attack_test_mode, self.transfer_attack_test_on_model_list,
                                          use_raw_nparray_data)
        # 攻击偏转能力评估
        attack_deflection_capability_evaluation(self.attacker_list, use_raw_nparray_data)
        logger.finish()

    def attack_adv_example_da_test_and_evaluation(self, use_raw_nparray_data=False):
        # 攻击偏转能力测试
        attack_adv_example_da_test(self.attacker_list, self.dataset_info, use_raw_nparray_data)
        # 攻击偏转能力评估
        attack_adv_example_da_evaluation(self.attacker_list, use_raw_nparray_data)
        logger.finish()

    def attack_full_test(self, use_img_file=True, use_raw_nparray_data=False):
        self.model_inference_capability_test_and_evaluation()
        self.only_build_adv()
        if not use_img_file and not use_raw_nparray_data:
            raise RuntimeError("[ Logic Error ] [ INIT TEST ] At least one format of data should be selected!")
        if use_img_file:
            self.attack_deflection_capability_test_and_evaluation(use_raw_nparray_data=False)
            self.attack_adv_example_da_test_and_evaluation(use_raw_nparray_data=False)
            # 模型综合能力测试结果分析
            model_security_synthetical_capability_evaluation(self.model_list, use_raw_nparray_data=False)
        if use_raw_nparray_data:
            self.attack_deflection_capability_test_and_evaluation(use_raw_nparray_data=True)
            self.attack_adv_example_da_test_and_evaluation(use_raw_nparray_data=True)
            # 模型综合能力测试结果分析
            model_security_synthetical_capability_evaluation(self.model_list, use_raw_nparray_data=True)

    def explore_attack_perturbation_test(self, use_raw_nparray_data=False):
        # 生成对抗样本
        explore_attack_perturbation(self.dataset_info, self.attacker_list, self.attacker_config, self.model_config, self.img_proc_config,
                                    self.explore_perturbation_config)
        # 测试
        explore_perturbation_attack_deflection_capability_test(self.attacker_list, self.model_config, self.img_proc_config, use_raw_nparray_data)
        explore_perturbation_attack_adv_example_da_test(self.attacker_list, self.dataset_info, use_raw_nparray_data)
        # 结果分析
        explore_perturbation_attack_capability_evaluation(self.attacker_list, use_raw_nparray_data)
        logger.finish()


# sys.excepthook = excepthook
