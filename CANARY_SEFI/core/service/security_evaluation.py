import sys
from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.batch_manager import batch_flag
from CANARY_SEFI.core.function.helper.batch_list_iterator import BatchListIterator
from CANARY_SEFI.core.function.helper.excepthook import excepthook
from CANARY_SEFI.core.function.init_dataset import init_dataset
from CANARY_SEFI.core.function.security_evaluation import model_capability_test, adv_img_build_and_evaluation, \
    attack_capability_test, explore_attack_perturbation, model_capability_evaluation, attack_capability_evaluation, \
    model_security_synthetical_capability_evaluation, explore_perturbation_attack_capability_test, \
    explore_perturbation_attack_capability_evaluation
from CANARY_SEFI.evaluator.logger.test_data_logger import log


class SecurityEvaluation:

    def __init__(self, dataset_name, dataset_size, dataset_seed=None):
        self.dataset_info = init_dataset(dataset_name, dataset_size, dataset_seed)
        print_logo()

    def only_build_adv(self, attacker_list, attacker_config, model_config, img_proc_config):
        # 生成对抗样本与对抗样本质量分析
        adv_img_build_and_evaluation(self.dataset_info, attacker_list, attacker_config, model_config, img_proc_config)
        log.finish()

    def attack_capability_test(self, adv_batch_id, attacker_list,
                    model_list, model_config, img_proc_config,
                    transfer_attack_test="NOT", transfer_attack_test_on_model_list=None):
        # 模型基线能力测试
        model_capability_test(self.dataset_info, model_list, model_config, img_proc_config)
        # 模型攻击测试
        attack_capability_test(adv_batch_id, attacker_list, model_config, img_proc_config, transfer_attack_test,
                               transfer_attack_test_on_model_list)
        log.finish()

    @staticmethod
    def attack_capability_evaluation(batch_id, model_list, attacker_list):
        # 模型能力测试结果分析
        model_capability_evaluation(batch_id, model_list)
        # 攻击测试结果分析
        attack_capability_evaluation(batch_id, attacker_list)
        log.finish()

    def attack_full_test(self,
                         attacker_list, attacker_config,
                         model_list, model_config, img_proc_config,
                         transfer_attack_test="NOT", transfer_attack_test_on_model_list=None):
        self.only_build_adv(attacker_list, attacker_config, model_config, img_proc_config)
        self.attack_capability_test(batch_flag.batch_id, attacker_list,
                    model_list, model_config, img_proc_config,
                    transfer_attack_test, transfer_attack_test_on_model_list)
        self.attack_capability_evaluation(batch_flag.batch_id, model_list, attacker_list)

        # 模型综合能力测试结果分析
        model_security_synthetical_capability_evaluation(batch_flag.batch_id, model_list)

    def explore_attack_perturbation_test(self, attacker_list, attacker_config,
                                         model_config, img_proc_config, explore_perturbation_config):
        # 生成对抗样本
        explore_attack_perturbation(self.dataset_info, attacker_list, attacker_config, model_config, img_proc_config,
                                    explore_perturbation_config)
        # 模型基线能力测试
        model_capability_test(self.dataset_info, BatchListIterator.get_singleton_model_list(attacker_list), model_config,
                              img_proc_config)
        # 模型能力测试结果分析
        model_capability_evaluation(batch_flag.batch_id, BatchListIterator.get_singleton_model_list(attacker_list))
        # 模型攻击测试
        explore_perturbation_attack_capability_test(batch_flag.batch_id, attacker_list, model_config, img_proc_config)
        # 攻击测试结果分析
        explore_perturbation_attack_capability_evaluation(batch_flag.batch_id, attacker_list)
        log.finish()


sys.excepthook = excepthook
