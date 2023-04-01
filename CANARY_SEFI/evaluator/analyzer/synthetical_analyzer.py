import re

import pandas as pd
import seaborn as sns
from colorama import Fore
from matplotlib import pyplot as plt

from CANARY_SEFI.handler.image_handler.plt_handler import figure_show_handler
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.indicator_data_handler import get_model_capability_indicator_data, \
    get_attack_deflection_capability_indicator_data_by_base_model, \
    get_attack_adv_example_da_indicator_data_by_base_model, add_model_security_synthetical_capability_log, \
    get_attack_deflection_capability_indicator_data_by_attack_name, \
    get_attack_adv_example_da_indicator_data_by_attack_name, get_attack_adv_example_cost_indicator_data_by_attack_name, \
    get_attack_adv_example_cost_indicator_data_by_base_model, add_attack_synthetical_capability_log
from CANARY_SEFI.evaluator.analyzer.analyzer_tools import calc_average


def adversarial_example_transfer_analyzer_log_handler(attack_deflection_capability_indicator_data, attack_name):
    adversarial_example_analyzer_log = {
        "T_MR": [], "T_AIAC": [], "T_ARTC": []
    }
    MR_data_map = {}
    AIAC_data_map = {}
    ARTC_data_map = {}

    def add_map(map, columns, index, data):
        columns = re.sub('\(.*?\)', '', columns)
        index = re.sub('\(.*?\)', '', index)
        if map.get(columns, None) is None:
            map[columns] = {}
        map[columns][index] = data

    transfer_flag = False
    for log in attack_deflection_capability_indicator_data:
        base_model, inference_model = log["base_model"], log["inference_model"]
        MR = str(log["MR"]).split('/')

        if base_model != inference_model:
            transfer_flag = True
            # 加入数据表
            add_map(MR_data_map, base_model, inference_model, float(MR[0]))
            add_map(AIAC_data_map, base_model, inference_model, float(log["AIAC"]))
            add_map(ARTC_data_map, base_model, inference_model, float(log["ARTC"]))

            adversarial_example_analyzer_log['T_MR'].append(float(MR[0]))
            adversarial_example_analyzer_log['T_AIAC'].append(float(log["AIAC"]))
            adversarial_example_analyzer_log['T_ARTC'].append(float(log["ARTC"]))

    if not transfer_flag:
        return None, None, None

    MR_data_df = pd.DataFrame(MR_data_map)
    AIAC_data_df = pd.DataFrame(AIAC_data_map)
    ARTC_data_df = pd.DataFrame(ARTC_data_map)

    fig = plt.figure(figsize=(24, 6.8), dpi=75)

    def show(data_df, title, subplot):
        data_df = data_df.reindex(data_df.mean(axis=0).sort_values().index, axis=1)
        data_df = data_df.reindex(data_df.mean(axis=1).sort_values().index, axis=0)

        # 设置标题、坐标轴标签及字体大小
        ax = fig.add_subplot(subplot)
        sns.heatmap(data_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax, annot_kws={"fontsize": 7.5})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor', size=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=35, ha='right', rotation_mode='anchor', size=10)
        ax.set_title('{} {} Heat Map'.format(attack_name, title), size="14")
        ax.set_xlabel('AEs Generation Based Model', size="13")
        ax.set_ylabel('AEs Transferred Target Model', size="13")

    show(MR_data_df, 'Misclassification Radio', 131)
    show(AIAC_data_df, 'Increase Adversarial-Class Confidence', 132)
    show(ARTC_data_df, 'Reduction True-class Confidence', 133)
    fig.tight_layout()
    figure_show_handler(fig, file_path="transfer_analyze_result/", file_name="transfer_heat_map")

    OTR_MR = calc_average(adversarial_example_analyzer_log["T_MR"])
    OTR_AIAC = calc_average(adversarial_example_analyzer_log["T_AIAC"])
    OTR_ARTC = calc_average(adversarial_example_analyzer_log["T_ARTC"])

    return OTR_MR, OTR_AIAC, OTR_ARTC

def adversarial_example_analyzer_log_handler(attack_deflection_capability_indicator_data,
                                             attack_adv_example_da_indicator_data,
                                             attack_adv_example_cost_indicator_data):
    adversarial_example_analyzer_log = {
        "MR": [], "TAS": [], "AIAC": [], "ARTC": [], "ACAMC_A": [], "ACAMC_T": [], "ACT": [], "AQN_F": [],
        "AQN_B": [], "AMD": [], "AED": [], "AED_HF": [], "AED_LF": [], "APCR": [], "ADMS": [], "ALMS": []
    }
    for log in attack_deflection_capability_indicator_data:
        # 排除转移测试
        if log["base_model"] == log["inference_model"]:
            MR = str(log["MR"]).split('/')
            adversarial_example_analyzer_log['MR'].append(float(MR[0]))
            if len(MR) == 2:
                adversarial_example_analyzer_log['TAS'].append(float(MR[1]))
            adversarial_example_analyzer_log['AIAC'].append(float(log["AIAC"]))
            adversarial_example_analyzer_log['ARTC'].append(float(log["ARTC"]))
            adversarial_example_analyzer_log['ACAMC_A'].append(float(log["ACAMC_A"]))
            adversarial_example_analyzer_log['ACAMC_T'].append(float(log["ACAMC_T"]))
    for log in attack_adv_example_da_indicator_data:
        adversarial_example_analyzer_log['AMD'].append(float(log["AMD"]))
        adversarial_example_analyzer_log['AED'].append(float(log["AED"]))
        adversarial_example_analyzer_log['AED_HF'].append(float(log["AED_HF"]))
        adversarial_example_analyzer_log['AED_LF'].append(float(log["AED_LF"]))
        adversarial_example_analyzer_log['APCR'].append(float(log["APCR"]))
        adversarial_example_analyzer_log['ADMS'].append(float(log["ADMS"]))
        adversarial_example_analyzer_log['ALMS'].append(float(log["ALMS"]))
    for log in attack_adv_example_cost_indicator_data:
        adversarial_example_analyzer_log['ACT'].append(float(log["ACT"]))
        adversarial_example_analyzer_log['AQN_F'].append(float(log["AQN_F"]))
        adversarial_example_analyzer_log['AQN_B'].append(float(log["AQN_B"]))

    MR = calc_average(adversarial_example_analyzer_log["MR"])
    TAS = calc_average(adversarial_example_analyzer_log["TAS"])
    AIAC = calc_average(adversarial_example_analyzer_log["AIAC"])
    ARTC = calc_average(adversarial_example_analyzer_log["ARTC"])
    ACAMC_A = calc_average(adversarial_example_analyzer_log["ACAMC_A"])
    ACAMC_T = calc_average(adversarial_example_analyzer_log["ACAMC_T"])
    ACT = calc_average(adversarial_example_analyzer_log["ACT"])
    AQN_F = calc_average(adversarial_example_analyzer_log["AQN_F"])
    AQN_B = calc_average(adversarial_example_analyzer_log["AQN_B"])
    AMD = calc_average(adversarial_example_analyzer_log["AMD"])
    AED = calc_average(adversarial_example_analyzer_log["AED"])
    AED_HF = calc_average(adversarial_example_analyzer_log["AED_HF"])
    AED_LF = calc_average(adversarial_example_analyzer_log["AED_LF"])
    APCR = calc_average(adversarial_example_analyzer_log["APCR"])
    ADMS = calc_average(adversarial_example_analyzer_log["ADMS"])
    ALMS = calc_average(adversarial_example_analyzer_log["ALMS"])

    return MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS

def model_security_synthetical_capability_analyzer_and_evaluation(model_name, use_raw_nparray_data=False):
    msg = "Analyze Model({})'s synthetical capability evaluation result.".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(model_name)
    if is_skip:
        return
    model_capability_indicator_data = get_model_capability_indicator_data(model_name)
    model_ACC = float(model_capability_indicator_data["clear_acc"])
    model_F1 = float(model_capability_indicator_data["clear_f1"])
    model_Conf = float(model_capability_indicator_data["clear_conf"])

    attack_deflection_capability_indicator_data = get_attack_deflection_capability_indicator_data_by_base_model(model_name)
    attack_adv_example_cost_indicator_data = get_attack_adv_example_cost_indicator_data_by_base_model(model_name)
    attack_adv_example_da_indicator_data = get_attack_adv_example_da_indicator_data_by_base_model(model_name)

    MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS =\
        adversarial_example_analyzer_log_handler(attack_deflection_capability_indicator_data,
                                                 attack_adv_example_da_indicator_data,
                                                 attack_adv_example_cost_indicator_data)

    test_adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    add_model_security_synthetical_capability_log(model_name, test_adv_example_file_type,
                                                  model_ACC, model_F1, model_Conf,
                                                  MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T,
                                                  ACT, AQN_F, AQN_B,
                                                  AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS)
    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)


def attack_synthetical_capability_analyzer_and_evaluation(attack_name, use_raw_nparray_data=False):
    msg = "Analyze Attack({})'s synthetical capability evaluation result.".format(attack_name)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(attack_name)
    if is_skip:
        return

    attack_deflection_capability_indicator_data = get_attack_deflection_capability_indicator_data_by_attack_name(attack_name)
    attack_adv_example_cost_indicator_data = get_attack_adv_example_cost_indicator_data_by_attack_name(attack_name)
    attack_adv_example_da_indicator_data = get_attack_adv_example_da_indicator_data_by_attack_name(attack_name)

    MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS =\
        adversarial_example_analyzer_log_handler(attack_deflection_capability_indicator_data,
                                                 attack_adv_example_da_indicator_data,
                                                 attack_adv_example_cost_indicator_data)
    OTR_MR, OTR_AIAC, OTR_ARTC =\
        adversarial_example_transfer_analyzer_log_handler(attack_deflection_capability_indicator_data, attack_name)

    model_analyzer_log = {
        "ACC": [], "F1": [], "Conf": []
    }
    for log in attack_adv_example_cost_indicator_data:
        model_capability_indicator_data = get_model_capability_indicator_data(log["base_model"])

        model_analyzer_log["ACC"].append(float(model_capability_indicator_data["clear_acc"]))
        model_analyzer_log["F1"].append(float(model_capability_indicator_data["clear_f1"]))
        model_analyzer_log["Conf"].append(float(model_capability_indicator_data["clear_conf"]))

    model_ACC = calc_average(model_analyzer_log["ACC"])
    model_F1 = calc_average(model_analyzer_log["F1"])
    model_Conf = calc_average(model_analyzer_log["Conf"])

    test_adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    add_attack_synthetical_capability_log(attack_name, test_adv_example_file_type,
                                          model_ACC, model_F1, model_Conf,
                                          MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, OTR_MR, OTR_AIAC, OTR_ARTC,
                                          ACT, AQN_F, AQN_B,
                                          AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS)
    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)
