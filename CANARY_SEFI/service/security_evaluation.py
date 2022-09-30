import sys
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.excepthook import excepthook
from CANARY_SEFI.core.function.init_dataset import init_dataset
from CANARY_SEFI.core.function.test_and_evaluation import explore_attack_perturbation, \
    explore_perturbation_attack_capability_evaluation, adv_example_generate, model_inference_capability_test, \
    model_inference_capability_evaluation, attack_deflection_capability_test, attack_deflection_capability_evaluation, \
    attack_adv_example_da_test, attack_adv_example_da_evaluation, model_security_synthetical_capability_evaluation, \
    explore_perturbation_attack_deflection_capability_test, explore_perturbation_attack_adv_example_da_test

logger = batch_manager.test_data_logger


class SecurityEvaluation:

    def __init__(self, dataset_name, dataset_size, dataset_seed=None):
        self.dataset_info = init_dataset(dataset_name, dataset_size, dataset_seed)

    def only_build_adv(self, attacker_list, attacker_config, model_config, img_proc_config):
        # 生成对抗样本与对抗样本质量分析
        adv_example_generate(self.dataset_info, attacker_list, attacker_config, model_config, img_proc_config)
        logger.finish()

    def model_inference_capability_test_and_evaluation(self, attacker_list, model_list, model_config, img_proc_config):
        # 模型推理能力测试
        model_inference_capability_test(self.dataset_info, model_list, model_config, img_proc_config)
        # 模型推理能力评估
        model_inference_capability_evaluation(attacker_list)
        logger.finish()

    def attack_deflection_capability_test_and_evaluation(self, attacker_list, model_config, img_proc_config,
                    transfer_attack_test="NOT", transfer_attack_test_on_model_list=None, use_raw_nparray_data=False):
        # 攻击偏转能力测试
        attack_deflection_capability_test(attacker_list, model_config, img_proc_config, transfer_attack_test,
                               transfer_attack_test_on_model_list, use_raw_nparray_data)
        # 攻击偏转能力评估
        attack_deflection_capability_evaluation(attacker_list, use_raw_nparray_data)
        logger.finish()

    def attack_adv_example_da_test_and_evaluation(self, attacker_list, use_raw_nparray_data=False):
        # 攻击偏转能力测试
        attack_adv_example_da_test(attacker_list, self.dataset_info, use_raw_nparray_data)
        # 攻击偏转能力评估
        attack_adv_example_da_evaluation(attacker_list, use_raw_nparray_data)
        logger.finish()

    def attack_full_test(self,
                         attacker_list, attacker_config,
                         model_list, model_config, img_proc_config,
                         transfer_attack_test="NOT", transfer_attack_test_on_model_list=None, use_raw_nparray_data=False):
        self.only_build_adv(attacker_list, attacker_config, model_config, img_proc_config)
        self.model_inference_capability_test_and_evaluation(attacker_list, model_list, model_config, img_proc_config)
        self.attack_deflection_capability_test_and_evaluation(attacker_list, model_config, img_proc_config,
                                                             transfer_attack_test,
                                                             transfer_attack_test_on_model_list,
                                                             use_raw_nparray_data)
        self.attack_adv_example_da_test_and_evaluation(attacker_list, use_raw_nparray_data)
        # 模型综合能力测试结果分析
        model_security_synthetical_capability_evaluation(model_list, use_raw_nparray_data=False)

    def explore_attack_perturbation_test(self, attacker_list, attacker_config,
                                         model_config, img_proc_config, explore_perturbation_config, use_raw_nparray_data=False):
        # 生成对抗样本
        explore_attack_perturbation(self.dataset_info, attacker_list, attacker_config, model_config, img_proc_config,
                                    explore_perturbation_config)
        # 测试
        explore_perturbation_attack_deflection_capability_test(attacker_list, model_config, img_proc_config, use_raw_nparray_data)
        explore_perturbation_attack_adv_example_da_test(attacker_list, self.dataset_info, use_raw_nparray_data)
        # 结果分析
        explore_perturbation_attack_capability_evaluation(attacker_list, use_raw_nparray_data)
        logger.finish()


sys.excepthook = excepthook
