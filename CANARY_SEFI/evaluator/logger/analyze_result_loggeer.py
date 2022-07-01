from colorama import Fore, Style

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_model_test_result_log(batch_id, model_name, clear_acc, clear_f1, clear_conf):
    sql = "INSERT INTO test_report_model_capability (batch_id,model_name,clear_acc,clear_f1,clear_conf) " \
          "VALUES ('{}', '{}', '{}', '{}', '{}')" \
        .format(str(batch_id), str(model_name), str(clear_acc), str(clear_f1), str(clear_conf))

    batch_id = log.insert_log(sql)
    if log.debug_log:
        print(Fore.CYAN + "[LOGGER] 已写入日志 {}模型的测评结果为: 准确率{} F1{} 置信度{}"
              .format(str(model_name), str(clear_acc), str(clear_f1), str(clear_conf)))
        print(Style.RESET_ALL)
    return batch_id


def add_attack_test_result_log(batch_id, atk_name, base_model, test_model_name, misclassification_ratio, average_increase_adversarial_class_confidence, average_reduction_true_class_confidence, average_cost_time):
    sql = "INSERT INTO test_report_attack_capability (batch_id,atk_name,base_model,test_model_name,misclassification_ratio," \
          "average_increase_adversarial_class_confidence,average_reduction_true_class_confidence,average_cost_time) " \
          "VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')" \
        .format(str(batch_id), str(atk_name), str(base_model), str(test_model_name), str(misclassification_ratio),
                str(average_increase_adversarial_class_confidence), str(average_reduction_true_class_confidence),
                str(average_cost_time))

    batch_id = log.insert_log(sql)
    if log.debug_log:
        print(Fore.CYAN + "[LOGGER] 已写入日志 {}模型的测评结果为: 准确率{} F1{} 置信度{}"
              .format(str(model_name), str(clear_acc), str(clear_f1), str(clear_conf)))
        print(Style.RESET_ALL)
    return batch_id


def add_adv_da_test_result_log(batch_id, atk_name, base_model, average_maximum_disturbance, average_euclidean_distortion, average_pixel_change_ratio, average_deep_metrics_similarity, average_low_level_metrics_similarity):
    sql = "INSERT INTO test_report_adv_da_capability (batch_id,atk_name,base_model,average_maximum_disturbance," \
          "average_euclidean_distortion,average_pixel_change_ratio,average_deep_metrics_similarity,average_low_level_metrics_similarity) " \
          "VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')" \
        .format(str(batch_id), str(atk_name), str(base_model), str(average_maximum_disturbance),
                str(average_euclidean_distortion), str(average_pixel_change_ratio),
                str(average_deep_metrics_similarity),str(average_low_level_metrics_similarity))

    batch_id = log.insert_log(sql)
    if log.debug_log:
        print(Fore.CYAN + "[LOGGER] 已写入日志 {}模型的测评结果为: 准确率{} F1{} 置信度{}"
              .format(str(model_name), str(clear_acc), str(clear_f1), str(clear_conf)))
        print(Style.RESET_ALL)
    return batch_id
