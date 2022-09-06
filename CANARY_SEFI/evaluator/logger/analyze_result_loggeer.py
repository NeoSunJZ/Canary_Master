from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_model_test_result_log(batch_id, model_name, clear_acc, clear_f1, clear_conf):
    sql = "INSERT INTO test_report_model_capability (batch_id,model_name,clear_acc,clear_f1,clear_conf) " \
          "VALUES ('{}', '{}', '{}', '{}', '{}')" \
        .format(str(batch_id), str(model_name), str(clear_acc), str(clear_f1), str(clear_conf))

    batch_id = log.insert_log(sql)
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 {}模型的测评结果为: 准确率:{} F1:{} 置信度:{}"\
            .format(str(model_name), str(clear_acc), str(clear_f1), str(clear_conf))
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
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
        msg = "[ LOGGER ] 已写入日志 基于 {} 模型 {} 攻击方法生成的对抗样本在 {} 模型上的测评结果为: " \
              "误分类率:{} 抗类平均置信偏离:{} 真实类平均置度偏离:{} 样本生成时间:{}"\
            .format(str(base_model), str(atk_name), str(test_model_name), str(misclassification_ratio),
                str(average_increase_adversarial_class_confidence), str(average_reduction_true_class_confidence),
                str(average_cost_time))
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
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
        msg = "[ LOGGER ] 已写入日志  基于 {} 模型 {} 攻击方法生成的对抗样本的测评结果为:" \
              "评价最大扰动距离:{} 平均欧式距离:{} 平均像素变化比例:{} 平均深度特征相似性:{} 平均低层特征相似性:{}"\
            .format(str(base_model), str(atk_name), str(average_maximum_disturbance),
                str(average_euclidean_distortion), str(average_pixel_change_ratio),
                str(average_deep_metrics_similarity),str(average_low_level_metrics_similarity))
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return batch_id
