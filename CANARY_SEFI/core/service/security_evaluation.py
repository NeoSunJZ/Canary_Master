from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.batch_list_iterator import BatchListIterator
from CANARY_SEFI.core.function.init_dataset import init_dataset
from CANARY_SEFI.core.function.security_evaluation import model_capability_test, adv_img_build_and_evaluation, \
    attack_capability_test, explore_attack_perturbation
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo
from CANARY_SEFI.evaluator.analyzer.test_analyzer import model_capability_evaluation, attack_capability_evaluation
from CANARY_SEFI.evaluator.logger.dataset_logger import add_dataset_log
from CANARY_SEFI.evaluator.logger.db_logger import log

class SecurityEvaluation:
    def __init__(self):
        self.full_adv_transfer_test = False

    def full_security_test(self, dataset_name, dataset_size, dataset_seed, attacker_list, attacker_config, model_config, img_proc_config):
        print_logo()
        # 构建数据集对象
        dataset_info = init_dataset(dataset_name, dataset_size, dataset_seed)
        # 模型基线能力测试
        model_capability_test(dataset_info, BatchListIterator.get_singleton_model_list(attacker_list), model_config, img_proc_config)
        # 模型能力测试结果分析
        model_capability_evaluation(batch_flag.batch_id)
        # 生成对抗样本与对抗样本质量分析
        adv_img_build_and_evaluation(dataset_info, attacker_list, attacker_config, model_config, img_proc_config)
        # 模型攻击测试
        attack_capability_test(batch_flag.batch_id, BatchListIterator.get_singleton_model_list(attacker_list), model_config, img_proc_config)
        # 攻击测试结果分析
        attack_capability_evaluation(batch_flag.batch_id)
        log.finish()

    def explore_attack_perturbation_test(self, dataset_name, dataset_size, dataset_seed, attacker_list, attacker_config, model_config, img_proc_config, explore_perturbation_config):
        print_logo()
        # 构建数据集对象
        dataset_info = init_dataset(dataset_name, dataset_size, dataset_seed)
        # 模型基线能力测试
        model_capability_test(dataset_info, BatchListIterator.get_singleton_model_list(attacker_list), model_config, img_proc_config)
        # 模型能力测试结果分析
        model_capability_evaluation(batch_flag.batch_id)
        # 生成对抗样本
        explore_attack_perturbation(dataset_info, attacker_list, attacker_config, model_config, img_proc_config, explore_perturbation_config)
        # 模型攻击测试
        attack_capability_test(batch_flag.batch_id, BatchListIterator.get_singleton_model_list(attacker_list), model_config, img_proc_config)
        # 攻击测试结果分析
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 3  攻击测试结果分析 ] <<----".format(batch_flag.batch_id))
        # explore_attack_perturbation_analyzer(batch_flag.batch_id)
        log.finish()
