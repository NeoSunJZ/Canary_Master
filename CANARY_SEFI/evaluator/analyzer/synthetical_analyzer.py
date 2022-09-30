from colorama import Fore

from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.evaluator.logger.indicator_data_handler import get_model_test_result_log, \
    get_attack_test_result_log_by_base_model, get_adv_da_test_result_log_by_base_model, \
    add_model_security_synthetical_capability_log
from CANARY_SEFI.handler.tools.analyzer_tools import calc_average


def model_security_synthetical_capability_analyzer_and_evaluation(batch_id, model_name):
    msg = "统计模型 {} 的综合安全能力测试结果".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(model_name)
    if is_skip:
        return
    model_inference_capability = get_model_test_result_log(batch_id, model_name)
    adversarial_example_analyzer_log = {
        "misclassification_ratio":[],
        "average_increase_adversarial_class_confidence": [],
        "average_reduction_true_class_confidence": [],
        "average_cost_time": [],
        "average_maximum_disturbance": [],
        "average_euclidean_distortion": [],
        "average_pixel_change_ratio": [],
        "average_deep_metrics_similarity": [],
        "average_low_level_metrics_similarity": []
    }
    adversarial_example_attack_capability = get_attack_test_result_log_by_base_model(batch_id, model_name)
    adversarial_example_disturbance_aware = get_adv_da_test_result_log_by_base_model(batch_id, model_name)
    for log in adversarial_example_attack_capability:
        # 排除转移测试
        if log["base_model"] == log["test_model_name"]:
            adversarial_example_analyzer_log['misclassification_ratio'].append(float(log["misclassification_ratio"]))
            adversarial_example_analyzer_log['average_increase_adversarial_class_confidence'].append(float(log["average_increase_adversarial_class_confidence"]))
            adversarial_example_analyzer_log['average_reduction_true_class_confidence'].append(float(log["average_reduction_true_class_confidence"]))
            adversarial_example_analyzer_log['average_cost_time'].append(float(log["average_cost_time"]))
    for log in adversarial_example_disturbance_aware:
        adversarial_example_analyzer_log['average_maximum_disturbance'].append(float(log["average_maximum_disturbance"]))
        adversarial_example_analyzer_log['average_euclidean_distortion'].append(float(log["average_euclidean_distortion"]))
        adversarial_example_analyzer_log['average_pixel_change_ratio'].append(float(log["average_pixel_change_ratio"]))
        adversarial_example_analyzer_log['average_deep_metrics_similarity'].append(float(log["average_deep_metrics_similarity"]))
        adversarial_example_analyzer_log['average_low_level_metrics_similarity'].append(float(log["average_low_level_metrics_similarity"]))

    model_clear_acc = float(model_inference_capability["clear_acc"])
    model_clear_f1 = float(model_inference_capability["clear_f1"])
    model_clear_conf = float(model_inference_capability["clear_conf"])
    model_misclassification_ratio = calc_average(adversarial_example_analyzer_log["misclassification_ratio"])
    model_average_increase_adversarial_class_confidence =calc_average(adversarial_example_analyzer_log["average_increase_adversarial_class_confidence"])
    model_average_reduction_true_class_confidence = calc_average(adversarial_example_analyzer_log["average_reduction_true_class_confidence"])
    model_average_cost_time = calc_average(adversarial_example_analyzer_log["average_cost_time"])
    model_average_maximum_disturbance = calc_average(adversarial_example_analyzer_log["average_maximum_disturbance"])
    model_average_euclidean_distortion =calc_average(adversarial_example_analyzer_log["average_euclidean_distortion"])
    model_average_pixel_change_ratio = calc_average(adversarial_example_analyzer_log["average_pixel_change_ratio"])
    model_average_deep_metrics_similarity = calc_average(adversarial_example_analyzer_log["average_deep_metrics_similarity"])
    model_average_low_level_metrics_similarity = calc_average(adversarial_example_analyzer_log["average_low_level_metrics_similarity"])

    add_model_security_synthetical_capability_log(batch_id, model_name, model_clear_acc, model_clear_f1, model_clear_conf,
                                                  model_misclassification_ratio,
                                                  model_average_increase_adversarial_class_confidence,
                                                  model_average_reduction_true_class_confidence,
                                                  model_average_cost_time,
                                                  model_average_maximum_disturbance,
                                                  model_average_euclidean_distortion,
                                                  model_average_pixel_change_ratio,
                                                  model_average_deep_metrics_similarity,
                                                  model_average_low_level_metrics_similarity)