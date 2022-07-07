import json

import numpy as np

from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.analyze_result_loggeer import add_model_test_result_log, add_attack_test_result_log, \
    add_adv_da_test_result_log
from CANARY_SEFI.evaluator.logger.adv_logger import find_batch_adv_log
from CANARY_SEFI.evaluator.logger.attack_logger import find_attack_log
from CANARY_SEFI.evaluator.logger.dataset_logger import find_img_log
from CANARY_SEFI.evaluator.logger.inference_logger import find_all_inference_log, find_inference_log
from CANARY_SEFI.handler.csv_handler.csv_io_handler import get_log_data_to_file
from sklearn.metrics import f1_score, accuracy_score


def model_capability_evaluation(batch_id):
    analyzer_log = {}
    all_inference_log = find_all_inference_log(DatasetType.NORMAL.value, batch_id)

    for inference_log in all_inference_log:
        model_name = inference_log[4]

        # 初始化
        if analyzer_log.get(model_name, None) is None:
            analyzer_log[model_name] = {
                "ori_labels": [], "inference_labels": [], "inference_confs": [],
            }
        # 获取预测标签
        analyzer_log[model_name]["inference_labels"].append(inference_log[5])
        # 获取真实标签
        ori_img_log = find_img_log(inference_log[1])[0]
        ori_label = ori_img_log[3]
        analyzer_log[model_name]["ori_labels"].append(ori_label)
        # 获取真实标签置信度
        inference_conf_array = np.array(eval(''.join(inference_log[6].replace(' ', ' ').replace(' ', ', '))))
        analyzer_log[model_name]["inference_confs"].append(inference_conf_array[ori_label])

    for model_name in analyzer_log:
        clear_acc = accuracy_score(analyzer_log[model_name]["ori_labels"], analyzer_log[model_name]["inference_labels"])
        clear_f1 = f1_score(analyzer_log[model_name]["ori_labels"], analyzer_log[model_name]["inference_labels"],
                            average='macro')
        clear_conf = sum(analyzer_log[model_name]["inference_confs"]) / len(analyzer_log[model_name]["inference_confs"])
        add_model_test_result_log(batch_id, model_name, clear_acc, clear_f1, clear_conf)


def attack_capability_evaluation(batch_id):
    analyzer_log = {}

    all_adv_example_log = find_batch_adv_log(batch_id)
    for adv_example_log in all_adv_example_log:
        attack_id = adv_example_log[2]

        # 初始化
        if analyzer_log.get(attack_id, None) is None:
            analyzer_log[attack_id] = {
                "attack_test_result": {}, "cost_time": [],
                "maximum_disturbance": [], "euclidean_distortion": [], "pixel_change_ratio": [],
                "deep_metrics_similarity": [], "low_level_metrics_similarity": []
            }

        adv_img_id = adv_example_log[0]
        ori_img_id = adv_example_log[4]

        adv_img_inference_log_list = find_inference_log(DatasetType.ADVERSARIAL_EXAMPLE.value, adv_img_id)
        ori_img_inference_log_list = find_inference_log(DatasetType.NORMAL.value, ori_img_id)

        ori_img_log = find_img_log(ori_img_id)[0]
        ori_label = ori_img_log[3]

        for adv_img_inference_log in adv_img_inference_log_list:
            for ori_img_inference_log in ori_img_inference_log_list:

                if adv_img_inference_log[4] == ori_img_inference_log[4]:  # 必须是同一模型的预测结果，否则没有参考价值

                    test_model_name = adv_img_inference_log[4]

                    if analyzer_log[attack_id]["attack_test_result"].get(test_model_name, None) is None:
                        analyzer_log[attack_id]["attack_test_result"][test_model_name] = {
                            "misclassification": [], "increase_adversarial_class_confidence": [],
                            "reduction_true_class_confidence": []
                        }

                    ori_inference_label = ori_img_inference_log[5]
                    adv_inference_label = adv_img_inference_log[5]

                    record = analyzer_log[attack_id]["attack_test_result"][test_model_name]

                    # 获取误分类数量(MR)
                    if ori_inference_label == ori_label:  # 必须是预测准确的才纳入计量
                        if adv_inference_label != ori_label:
                            record["misclassification"].append(1)
                        else:
                            record["misclassification"].append(0)

                    # 获取置信偏移
                    adv_inference_conf_array = np.array(eval(', '.join(adv_img_inference_log[6].split())))
                    ori_inference_conf_array = np.array(eval(', '.join(ori_img_inference_log[6].split())))

                    record["increase_adversarial_class_confidence"]\
                        .append(adv_inference_conf_array[adv_inference_label] - ori_inference_conf_array[adv_inference_label])

                    record["reduction_true_class_confidence"]\
                        .append(ori_inference_conf_array[ori_label] - adv_inference_conf_array[ori_label])

        record = analyzer_log[attack_id]
        # 获取消耗时间
        record["cost_time"].append(adv_example_log[3])

        # 获取对抗样本扰动测试项目
        record["maximum_disturbance"].append(float(adv_example_log[6]))
        record["euclidean_distortion"].append(float(adv_example_log[7]))
        record["pixel_change_ratio"].append(float(adv_example_log[8]))
        record["deep_metrics_similarity"].append(float(adv_example_log[9]))
        record["low_level_metrics_similarity"].append(float(adv_example_log[10]))

    for attack_id in analyzer_log:
        attack_log = find_attack_log(attack_id)[0]
        atk_name = attack_log[2]
        base_model = attack_log[3]

        record = analyzer_log[attack_id]
        average_maximum_disturbance = sum(record["maximum_disturbance"]) / len(record["maximum_disturbance"])
        average_euclidean_distortion = sum(record["euclidean_distortion"]) / len(record["euclidean_distortion"])
        average_pixel_change_ratio = sum(record["pixel_change_ratio"]) / len(record["pixel_change_ratio"])
        average_deep_metrics_similarity = sum(record["deep_metrics_similarity"]) / len(record["deep_metrics_similarity"])
        average_low_level_metrics_similarity = sum(record["low_level_metrics_similarity"]) / len(record["low_level_metrics_similarity"])

        # 写入日志
        add_adv_da_test_result_log(batch_id, atk_name, base_model, average_maximum_disturbance,
                                   average_euclidean_distortion, average_pixel_change_ratio,
                                   average_deep_metrics_similarity, average_low_level_metrics_similarity)

        average_cost_time = sum(record["cost_time"]) / len(record["cost_time"])

        attack_test_result = analyzer_log[attack_id]["attack_test_result"]

        for test_model_name in attack_test_result:

            record = attack_test_result[test_model_name]

            valid_sample_count = len(record["misclassification"])
            if valid_sample_count == 0:
                misclassification_ratio = 0
            else:
                misclassification_ratio = sum(record["misclassification"]) / valid_sample_count

            average_increase_adversarial_class_confidence = sum(record["increase_adversarial_class_confidence"]) / len(record["increase_adversarial_class_confidence"])
            average_reduction_true_class_confidence = sum(record["reduction_true_class_confidence"]) / len(record["reduction_true_class_confidence"])
            # 写入日志
            add_attack_test_result_log(batch_id, atk_name, base_model, test_model_name, misclassification_ratio,
                                       average_increase_adversarial_class_confidence,
                                       average_reduction_true_class_confidence, average_cost_time)
