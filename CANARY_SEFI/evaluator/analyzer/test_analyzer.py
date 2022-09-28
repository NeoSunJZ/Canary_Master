import numpy as np
from colorama import Fore
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_logger import find_adv_log_by_attack_id
from CANARY_SEFI.evaluator.logger.analyze_result_loggeer import add_model_test_result_log, add_attack_test_result_log, \
    add_adv_da_test_result_log
from CANARY_SEFI.evaluator.logger.attack_logger import find_attack_log_by_name_and_base_model
from CANARY_SEFI.evaluator.logger.dataset_logger import find_img_log
from CANARY_SEFI.evaluator.logger.inference_logger import find_inference_log_by_model_name, \
    find_inference_log_by_img_id, read_conf_array
from sklearn.metrics import f1_score, accuracy_score

from CANARY_SEFI.handler.tools.analyzer_tools import calc_average


def model_capability_analyzer_and_evaluation(batch_id, model_name):
    msg = "统计模型 {} 的能力测试结果".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(model_name)
    if is_skip:
        return
    all_inference_log = find_inference_log_by_model_name(DatasetType.NORMAL.value, batch_id, model_name)
    # 初始化
    analyzer_log = {
        "ori_labels": [], "inference_labels": [], "inference_confs": [],
    }
    for inference_log in all_inference_log:
        # 获取预测标签
        analyzer_log["inference_labels"].append(inference_log["inference_img_label"])
        # 获取真实标签
        ori_img_log = find_img_log(inference_log["img_id"])
        ori_label = ori_img_log["ori_img_label"]
        analyzer_log["ori_labels"].append(ori_label)
        # 获取真实标签置信度
        analyzer_log["inference_confs"].append(inference_log["inference_img_conf_array"][ori_label])

    msg = "计算模型 {} 的能力测试结果".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    clear_acc = accuracy_score(analyzer_log["ori_labels"], analyzer_log["inference_labels"])
    clear_f1 = f1_score(analyzer_log["ori_labels"], analyzer_log["inference_labels"],
                        average='macro')
    clear_conf = calc_average(analyzer_log["inference_confs"])
    add_model_test_result_log(batch_id, model_name, clear_acc, clear_f1, clear_conf)
    # 增加计数
    global_system_log.update_completed_num(1)
    global_system_log.update_finish_status(True)


def attack_capability_analyzer_and_evaluation(batch_id, atk_name, base_model):
    msg = "统计攻击方法 {} (基于 {} 模型) 生成的对抗样本质量与攻击能力测试结果".format(atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model)
    if is_skip:
        return
    attack_info = find_attack_log_by_name_and_base_model(batch_id, atk_name, base_model)
    attack_test_analyzer_and_evaluation_handler(batch_id, attack_info)


def attack_test_analyzer_and_evaluation_handler(batch_id, attack_info):
    all_adv_example_log = find_adv_log_by_attack_id(batch_id, attack_info['attack_id'])
    analyzer_log = {
        "attack_test_result": {}, "cost_time": [],
        "maximum_disturbance": [], "euclidean_distortion": [], "pixel_change_ratio": [],
        "deep_metrics_similarity": [], "low_level_metrics_similarity": []
    }

    for adv_example_log in all_adv_example_log:
        adv_img_id = adv_example_log["adv_img_id"]  # 对抗样本ID
        ori_img_id = adv_example_log["ori_img_id"]  # 原始图片ID

        # 对抗样本的预测记录
        adv_img_inference_log_list = find_inference_log_by_img_id(DatasetType.ADVERSARIAL_EXAMPLE.value, adv_img_id)
        # 原始图片的预测记录
        ori_img_inference_log_list = find_inference_log_by_img_id(DatasetType.NORMAL.value, ori_img_id)

        ori_img_log = find_img_log(ori_img_id)
        ori_label = ori_img_log["ori_img_label"] # 原始图片的标签

        # 确定对抗样本有效性(生成模型必须是预测准确的)
        is_valid = True
        for ori_img_inference_log in ori_img_inference_log_list:
            if ori_img_inference_log["inference_model"] == attack_info['base_model']:
                is_valid = ori_img_inference_log["inference_img_label"] == ori_label
        if not is_valid:
            msg = "adv_img_id {} 为无效样本，已经抛弃".format(adv_img_id)
            reporter.console_log(msg, Fore.GREEN, save_db=False, send_msg=False, show_batch=True, show_step_sequence=True)
            continue

        for adv_img_inference_log in adv_img_inference_log_list:
            for ori_img_inference_log in ori_img_inference_log_list:
                if adv_img_inference_log["inference_model"] == ori_img_inference_log["inference_model"]:
                    # 寻找两个记录中模型一致的(在同一个模型上进行的测试)

                    ori_inference_label = ori_img_inference_log["inference_img_label"] # 原始图片预测的标签
                    adv_inference_label = adv_img_inference_log["inference_img_label"] # 对抗图片预测的标签

                    if ori_inference_label == ori_label: # 原始图片在该测试模型上预测准确(不准确的直接无效处理)
                        test_model_name = adv_img_inference_log["inference_model"]
                        # 初始化每个模型的测试结果统计
                        if analyzer_log["attack_test_result"].get(test_model_name, None) is None:
                            analyzer_log["attack_test_result"][test_model_name] = {
                                "misclassification": [], "increase_adversarial_class_confidence": [],
                                "reduction_true_class_confidence": []
                            }
                        record = analyzer_log["attack_test_result"][test_model_name]
                        # 获取误分类数量(MR)
                        if adv_inference_label != ori_label:
                            record["misclassification"].append(1)
                        else:
                            record["misclassification"].append(0)
                        # 获取置信偏移
                        record["increase_adversarial_class_confidence"] \
                            .append(adv_img_inference_log["inference_img_conf_array"][adv_inference_label] -
                                    ori_img_inference_log["inference_img_conf_array"][adv_inference_label])
                        record["reduction_true_class_confidence"] \
                            .append(ori_img_inference_log["inference_img_conf_array"][ori_label] -
                                    adv_img_inference_log["inference_img_conf_array"][ori_label])
        # 获取消耗时间
        analyzer_log["cost_time"].append(adv_example_log["cost_time"])
        # 获取对抗样本扰动测试项目
        analyzer_log["maximum_disturbance"].append(float(adv_example_log["adv_img_maximum_disturbance"]))
        analyzer_log["euclidean_distortion"].append(float(adv_example_log["adv_img_euclidean_distortion"]))
        analyzer_log["pixel_change_ratio"].append(float(adv_example_log["adv_img_pixel_change_ratio"]))
        analyzer_log["deep_metrics_similarity"].append(float(adv_example_log["adv_img_deep_metrics_similarity"]))
        analyzer_log["low_level_metrics_similarity"].append(
            float(adv_example_log["adv_img_low_level_metrics_similarity"]))

    msg = "计算对抗方法 {} (基于 {} 模型) 生成的对抗样本的质量测评结果".format(attack_info['atk_name'], attack_info['base_model'])
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    average_maximum_disturbance = calc_average(analyzer_log["maximum_disturbance"])
    average_euclidean_distortion = calc_average(analyzer_log["euclidean_distortion"])
    average_pixel_change_ratio = calc_average(analyzer_log["pixel_change_ratio"])
    average_deep_metrics_similarity = calc_average(analyzer_log["deep_metrics_similarity"])
    average_low_level_metrics_similarity = calc_average(analyzer_log["low_level_metrics_similarity"])

    # 写入日志
    add_adv_da_test_result_log(batch_id, attack_info['atk_name'], attack_info['base_model'], average_maximum_disturbance,
                               average_euclidean_distortion, average_pixel_change_ratio,
                               average_deep_metrics_similarity, average_low_level_metrics_similarity,
                               atk_perturbation_budget=attack_info['atk_perturbation_budget'])

    msg = "计算对抗方法 {} (基于 {} 模型) 生成的对抗样本的攻击能力测评结果".format(attack_info['atk_name'], attack_info['base_model'])
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)
    average_cost_time = calc_average(analyzer_log["cost_time"])
    for test_model_name in analyzer_log["attack_test_result"]:
        record = analyzer_log["attack_test_result"][test_model_name]

        misclassification_ratio = calc_average(record["misclassification"])
        average_increase_adversarial_class_confidence = calc_average(record["increase_adversarial_class_confidence"])
        average_reduction_true_class_confidence = calc_average(record["reduction_true_class_confidence"])
        # 写入日志
        add_attack_test_result_log(batch_id, attack_info['atk_name'], attack_info['base_model'], test_model_name,
                                   misclassification_ratio,
                                   average_increase_adversarial_class_confidence,
                                   average_reduction_true_class_confidence, average_cost_time,
                                   atk_perturbation_budget=attack_info['atk_perturbation_budget'])
    # 增加计数
    global_system_log.update_completed_num(1)
    global_system_log.update_finish_status(True)
