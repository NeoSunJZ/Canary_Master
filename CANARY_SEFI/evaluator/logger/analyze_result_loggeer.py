from colorama import Fore
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_model_test_result_log(batch_id, model_name, clear_acc, clear_f1, clear_conf):
    sql = "INSERT INTO test_report_model_capability (batch_id, model_name, clear_acc, clear_f1, clear_conf) " \
          "VALUES (?,?,?,?,?)"
    batch_id = log.insert_log(sql, (str(batch_id), str(model_name), str(clear_acc), str(clear_f1), str(clear_conf)))
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 {} 模型的测评结果为: 准确率:{} F1:{} 置信度:{}"\
            .format(str(model_name), str(clear_acc), str(clear_f1), str(clear_conf))
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return batch_id


def get_model_test_result_log(batch_id, model_name):
    sql = " SELECT * FROM test_report_model_capability WHERE batch_id = ? AND model_name = ?"
    return log.query_log(sql, (batch_id, model_name))


def add_attack_test_result_log(batch_id, atk_name, base_model, test_model_name, misclassification_ratio,
                               average_increase_adversarial_class_confidence, average_reduction_true_class_confidence,
                               average_cost_time, atk_perturbation_budget=None):
    sql = "INSERT INTO test_report_attack_capability (batch_id, atk_name, base_model, atk_perturbation_budget, " \
          "test_model_name, misclassification_ratio, average_increase_adversarial_class_confidence, " \
          "average_reduction_true_class_confidence, average_cost_time) " \
          "VALUES (?,?,?,?,?,?,?,?,?)"
    batch_id = log.insert_log(sql, (str(batch_id), str(atk_name), str(base_model), str(atk_perturbation_budget),
                                    str(test_model_name), str(misclassification_ratio),
                                    str(average_increase_adversarial_class_confidence),
                                    str(average_reduction_true_class_confidence),str(average_cost_time)))
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 攻击方法 {} (基于 {} 模型) 生成的对抗样本在 {} 模型上的测评结果为: " \
              "误分类率:{} 抗类平均置信偏离:{} 真实类平均置度偏离:{} 样本生成时间:{}"\
            .format(str(atk_name), str(base_model), str(test_model_name), str(misclassification_ratio),
                str(average_increase_adversarial_class_confidence), str(average_reduction_true_class_confidence),
                str(average_cost_time))
        if atk_perturbation_budget is not None:
            msg += "(最佳扰动探索，当前探索扰动:{})".format(atk_perturbation_budget)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return batch_id


def get_attack_test_result_log_by_base_model(batch_id, base_model):
    sql = " SELECT * FROM test_report_attack_capability WHERE batch_id = ? AND base_model = ? "
    return log.query_logs(sql, (batch_id, base_model))


def get_attack_test_result_log_by_attack_name(batch_id, atk_name):
    sql = " SELECT * FROM test_report_attack_capability WHERE batch_id = ? AND atk_name = ? "
    return log.query_logs(sql, (batch_id, atk_name))


def add_adv_da_test_result_log(batch_id, atk_name, base_model, average_maximum_disturbance,
                               average_euclidean_distortion, average_pixel_change_ratio,
                               average_deep_metrics_similarity, average_low_level_metrics_similarity,
                               atk_perturbation_budget=None):
    sql = "INSERT INTO test_report_adv_da_capability (batch_id,atk_name,base_model,atk_perturbation_budget," \
          "average_maximum_disturbance,average_euclidean_distortion,average_pixel_change_ratio," \
          "average_deep_metrics_similarity,average_low_level_metrics_similarity) " \
          "VALUES (?,?,?,?,?,?,?,?,?)"
    batch_id = log.insert_log(sql, (str(batch_id), str(atk_name), str(base_model), str(atk_perturbation_budget),
                                    str(average_maximum_disturbance),str(average_euclidean_distortion),
                                    str(average_pixel_change_ratio),
                                    str(average_deep_metrics_similarity),str(average_low_level_metrics_similarity)))
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 攻击方法 {} (基于 {} 模型) 生成的对抗样本的测评结果为:" \
              "评价最大扰动距离:{} 平均欧式距离:{} 平均像素变化比例:{} 平均深度特征相似性:{} 平均低层特征相似性:{}"\
            .format(str(atk_name), str(base_model), str(average_maximum_disturbance),
                str(average_euclidean_distortion), str(average_pixel_change_ratio),
                str(average_deep_metrics_similarity),str(average_low_level_metrics_similarity))
        if atk_perturbation_budget is not None:
            msg += "(最佳扰动探索，当前探索扰动:{})".format(atk_perturbation_budget)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return batch_id


def get_adv_da_test_result_log_by_base_model(batch_id, base_model):
    sql = " SELECT * FROM test_report_adv_da_capability WHERE batch_id = ? AND base_model = ? "
    return log.query_logs(sql, (batch_id, base_model))


def get_adv_da_test_result_log_by_attack_name(batch_id, atk_name):
    sql = " SELECT * FROM test_report_adv_da_capability WHERE batch_id = ? AND atk_name = ? "
    return log.query_logs(sql, (batch_id, atk_name))


def get_explore_perturbation_result_by_attack_name_and_base_model(batch_id, atk_name, base_model):
    sql = " SELECT test_report_attack_capability.*,test_report_adv_da_capability.* " \
          " FROM test_report_attack_capability,test_report_adv_da_capability " \
          " WHERE test_report_attack_capability.batch_id = test_report_adv_da_capability.batch_id " \
          " AND test_report_attack_capability.atk_name = test_report_adv_da_capability.atk_name " \
          " AND test_report_attack_capability.base_model = test_report_adv_da_capability.base_model " \
          " AND test_report_attack_capability.atk_perturbation_budget = test_report_adv_da_capability.atk_perturbation_budget " \
          " AND test_report_attack_capability.batch_id = ? " \
          " AND test_report_attack_capability.atk_name = ? " \
          " AND test_report_attack_capability.base_model = ? "
    return log.query_logs(sql, (batch_id, atk_name, base_model))


def add_model_security_synthetical_capability_log(batch_id, model_name, model_clear_acc, model_clear_f1, model_clear_conf,
                                                  model_misclassification_ratio,
                                                  model_average_increase_adversarial_class_confidence,
                                                  model_average_reduction_true_class_confidence,
                                                  model_average_cost_time,
                                                  model_average_maximum_disturbance,
                                                  model_average_euclidean_distortion,
                                                  model_average_pixel_change_ratio,
                                                  model_average_deep_metrics_similarity,
                                                  model_average_low_level_metrics_similarity):
    sql = "INSERT INTO model_security_synthetical_capability (batch_id,model_name," \
          "clear_acc,clear_f1,clear_conf,misclassification_ratio," \
          "average_increase_adversarial_class_confidence,average_reduction_true_class_confidence," \
          "average_cost_time,average_maximum_disturbance,average_euclidean_distortion,average_pixel_change_ratio," \
          "average_deep_metrics_similarity,average_low_level_metrics_similarity) " \
          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    batch_id = log.insert_log(sql, (batch_id, model_name, model_clear_acc, model_clear_f1, model_clear_conf,
                                    model_misclassification_ratio,
                                    model_average_increase_adversarial_class_confidence,
                                    model_average_reduction_true_class_confidence,
                                    model_average_cost_time, model_average_maximum_disturbance,
                                    model_average_euclidean_distortion, model_average_pixel_change_ratio,
                                    model_average_deep_metrics_similarity, model_average_low_level_metrics_similarity))
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 {} 模型的测评结果为: 准确率:{} F1:{} 置信度:{}" \
              "受攻击后 误分类率:{} 抗类平均置信偏离:{} 真实类平均置度偏离:{} 攻击样本生成时间:{}" \
              "受攻击强度 最大扰动距离:{} 平均欧式距离:{} 平均像素变化比例:{} 平均深度特征相似性:{} 平均低层特征相似性:{}" \
            .format(model_name, model_clear_acc, model_clear_f1, model_clear_conf,model_misclassification_ratio,
                    model_average_increase_adversarial_class_confidence,
                    model_average_reduction_true_class_confidence,
                    model_average_cost_time,
                    model_average_maximum_disturbance, model_average_euclidean_distortion,
                    model_average_pixel_change_ratio,
                    model_average_deep_metrics_similarity, model_average_low_level_metrics_similarity)

        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return batch_id

def get_model_security_synthetical_capability_log(batch_id, model_name):
    sql = " SELECT * FROM model_security_synthetical_capability WHERE batch_id = ? AND model_name = ? "
    return log.query_log(sql, (batch_id, model_name))