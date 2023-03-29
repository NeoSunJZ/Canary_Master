import math

import numpy as np
from colorama import Fore
from sklearn.metrics import accuracy_score

from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id
from CANARY_SEFI.evaluator.logger.attack_info_handler import find_attack_log_by_name_and_base_model
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.evaluator.logger.indicator_data_handler import save_defense_normal_effectiveness_data, \
    save_defense_adv_effectiveness_data
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import get_inference_test_data_by_model_name, \
    get_inference_test_data_by_img_id, get_inference_test_data_by_img_id_and_model
from CANARY_SEFI.task_manager import task_manager


def JS_Divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    M = (p + q) / 2
    return 0.5 * np.sum(p * np.log(p / M)) + 0.5 * np.sum(q * np.log(q / M))


def analyzer_log_handler(base_model, dataset_type):
    base_model_inference_log = get_inference_test_data_by_model_name(base_model, dataset_type)
    analyzer_log = {
        "ori_labels": [], "inference_labels": [], "inference_confs": [],
    }
    for inference_log in base_model_inference_log:
        # 获取预测标签
        analyzer_log["inference_labels"].append(inference_log["inference_img_label"])
        # 获取真实标签
        ori_img_log = find_img_log_by_id(inference_log["img_id"])
        ori_label = ori_img_log["ori_img_label"]
        analyzer_log["ori_labels"].append(ori_label)
        # 获取真实标签置信度
        analyzer_log["inference_confs"].append(inference_log["inference_img_conf_array"])

    return analyzer_log


def defense_normal_effectiveness_analyzer_and_evaluation(base_normal_analyzer_log, dataset_info, defense_name,
                                                         base_model):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}*)'s effectiveness with normal test result".format(
        defense_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
    # base_normal_analyzer_log = analyzer_log_handler(base_model, DatasetType.NORMAL.value)
    defense_normal_analyzer_log = analyzer_log_handler(base_model + '_' + defense_name, DatasetType.NORMAL.value)

    DFAcc = accuracy_score(defense_normal_analyzer_log["ori_labels"], defense_normal_analyzer_log["inference_labels"])
    Acc = accuracy_score(base_normal_analyzer_log["ori_labels"], base_normal_analyzer_log["inference_labels"])
    CRR, CSR, ConV, COS, num = 0, 0, 0, 0, 0
    for i in range(dataset_info.dataset_size):
        FXi = base_normal_analyzer_log["inference_labels"][i]
        FDXi = defense_normal_analyzer_log["inference_labels"][i]
        yi = base_normal_analyzer_log["ori_labels"][i]
        PXi = base_normal_analyzer_log["inference_confs"][i]
        PDXi = defense_normal_analyzer_log["inference_confs"][i]
        if (FXi != yi) and (FDXi == yi):
            CRR += 1
        elif (FXi == yi) and (FDXi != yi):
            CSR += 1
        elif (FXi == yi) and (FDXi == yi):
            num += 1
            ConV += math.fabs(PDXi[yi] - PXi[yi])
            COS += JS_Divergence(PXi, PDXi)

    CAV = DFAcc - Acc
    RRSR = CRR  # 原本为CRR/CSR，但是CSR可能为0，所以在考虑如何转变
    ConV = ConV / num
    COS = COS / num

    save_defense_normal_effectiveness_data(base_model + '_' + defense_name, CAV, RRSR, ConV, COS)
    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)


def defense_adv_effectiveness_analyzer_and_evaluation(atk_name, defense_name, base_model,
                                                      use_raw_nparray_data=False):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}*)'s effectiveness with adv example test result".format(
        defense_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
    normal_adv_inference_acc = defense_adv_effectiveness_analyzer_and_evaluation_handler(atk_name, base_model,
                                                                                         base_model,
                                                                                         use_raw_nparray_data)
    normal_trasfer_defense_acc = defense_adv_effectiveness_analyzer_and_evaluation_handler(atk_name, base_model,
                                                                                           base_model + "_" + defense_name,
                                                                                           use_raw_nparray_data)
    defense_adv_inference_acc = defense_adv_effectiveness_analyzer_and_evaluation_handler(atk_name,
                                                                                          base_model + "_" + defense_name,
                                                                                          base_model + "_" + defense_name,
                                                                                          use_raw_nparray_data)
    DCAV = defense_adv_inference_acc - normal_adv_inference_acc
    TCAV = normal_trasfer_defense_acc - normal_adv_inference_acc

    save_defense_adv_effectiveness_data(model_name=base_model + "_" + defense_name, attack_name=atk_name, DCAV=DCAV,
                                        TCAV=TCAV)
    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)


def defense_adv_effectiveness_analyzer_and_evaluation_handler(atk_name, base_model, inference_model,
                                                              use_raw_nparray_data=False):
    base_attack_info = find_attack_log_by_name_and_base_model(atk_name, base_model)
    all_adv_example_file_log = find_adv_example_file_logs_by_attack_id(base_attack_info['attack_id'])
    adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value
    analyzer_log = {
        "ori_labels": [], "inference_labels": [],
    }
    for adv_example_file_log in all_adv_example_file_log:
        ori_img_id = adv_example_file_log["ori_img_id"]  # 原始图片ID(干净样本O)
        adv_img_file_id = adv_example_file_log["adv_img_file_id"]  # 对抗样本ID(干净样本O通过在M1上的攻击产生对抗样本A)

        ori_img_log = find_img_log_by_id(ori_img_id)  # 原始图片记录
        ori_label = ori_img_log["ori_img_label"]  # 原始图片的标签
        analyzer_log["ori_labels"].append(ori_label)
        adv_img_inference_log = get_inference_test_data_by_img_id_and_model(adv_img_file_id, adv_example_file_type,
                                                                            inference_model)
        inference_label = adv_img_inference_log[0]["inference_img_label"]

        analyzer_log["inference_labels"].append(inference_label)

    acc = accuracy_score(analyzer_log["ori_labels"], analyzer_log["inference_labels"])
    return acc
