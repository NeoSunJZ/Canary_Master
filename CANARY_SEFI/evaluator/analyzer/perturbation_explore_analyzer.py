from colorama import Fore
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.analyzer.test_analyzer import attack_test_analyzer_and_evaluation_handler
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_log_by_attack_id
from CANARY_SEFI.evaluator.logger.indicator_data_handler import add_attack_test_result_log, add_adv_da_test_result_log
from CANARY_SEFI.evaluator.logger.attack_info_handler import find_attack_log_by_name_and_base_model
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import find_inference_log_by_img_id
from CANARY_SEFI.handler.tools.analyzer_tools import calc_average


def perturbation_explore_analyzer_and_evaluation(batch_id, atk_name, base_model):

    msg = "统计攻击方法 {} (基于 {} 模型) 生成的对抗样本扰动探索结果".format(atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model)
    if is_skip:
        return

    attack_logs = find_attack_log_by_name_and_base_model(batch_id, atk_name, base_model, explore_perturbation_mode=True)
    for attack_info in attack_logs:
        attack_test_analyzer_and_evaluation_handler(batch_id, attack_info)