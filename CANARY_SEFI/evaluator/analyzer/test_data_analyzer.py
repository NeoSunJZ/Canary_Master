from colorama import Fore

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_da_test_data_handler import find_adv_example_da_test_data_by_id_and_type
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.evaluator.logger.indicator_data_handler import save_attack_deflection_capability_indicator_data, \
    save_attack_adv_example_da_indicator_data, save_model_capability_indicator_data
from CANARY_SEFI.evaluator.logger.attack_info_handler import find_attack_log_by_name_and_base_model
from sklearn.metrics import f1_score, accuracy_score

from CANARY_SEFI.evaluator.logger.inference_test_data_handler import get_inference_test_data_by_model_name, \
    get_inference_test_data_by_img_id
from CANARY_SEFI.handler.tools.analyzer_tools import calc_average


def model_inference_capability_analyzer_and_evaluation(model_name):
    msg = "Analyzing and Evaluating Model({})'s inference capability test result".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(model_name)
    if is_skip:
        return
    all_inference_log = get_inference_test_data_by_model_name(model_name, DatasetType.NORMAL.value)
    # 初始化
    analyzer_log = {
        "ori_labels": [], "inference_labels": [], "inference_confs": [],
    }
    for inference_log in all_inference_log:
        # 获取预测标签
        analyzer_log["inference_labels"].append(inference_log["inference_img_label"])
        # 获取真实标签
        ori_img_log = find_img_log_by_id(inference_log["img_id"])
        ori_label = ori_img_log["ori_img_label"]
        analyzer_log["ori_labels"].append(ori_label)
        # 获取真实标签置信度
        analyzer_log["inference_confs"].append(inference_log["inference_img_conf_array"][ori_label])

    clear_acc = accuracy_score(analyzer_log["ori_labels"], analyzer_log["inference_labels"])
    clear_f1 = f1_score(analyzer_log["ori_labels"], analyzer_log["inference_labels"],
                        average='macro')
    clear_conf = calc_average(analyzer_log["inference_confs"])
    save_model_capability_indicator_data(model_name, clear_acc, clear_f1, clear_conf)
    # 增加计数
    batch_manager.sys_log_logger.update_completed_num(1)
    batch_manager.sys_log_logger.update_finish_status(True)


def attack_deflection_capability_analyzer_and_evaluation(atk_name, base_model, use_raw_nparray_data=False):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}*)'s deflection capability test result".format(atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model)
    if is_skip:
        return
    attack_info = find_attack_log_by_name_and_base_model(atk_name, base_model)
    attack_deflection_capability_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data)


def attack_deflection_capability_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data=False):
    all_adv_example_log = find_adv_example_file_logs_by_attack_id(attack_info['attack_id'])

    attack_test_result = {}

    adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    for adv_example_log in all_adv_example_log:
        adv_img_file_id = adv_example_log["adv_img_file_id"]  # 对抗样本ID
        ori_img_id = adv_example_log["ori_img_id"]  # 原始图片ID

        # 对抗样本的预测记录
        adv_img_inference_log_list = get_inference_test_data_by_img_id(adv_img_file_id, adv_example_file_type)
        # 原始图片的预测记录
        ori_img_inference_log_list = get_inference_test_data_by_img_id(ori_img_id, DatasetType.NORMAL.value)

        ori_img_log = find_img_log_by_id(ori_img_id)
        ori_label = ori_img_log["ori_img_label"] # 原始图片的标签

        # 确定对抗样本有效性(生成模型必须是预测准确的)
        is_valid = True
        for ori_img_inference_log in ori_img_inference_log_list:
            if ori_img_inference_log["inference_model"] == attack_info['base_model']:
                is_valid = ori_img_inference_log["inference_img_label"] == ori_label
        if not is_valid:
            msg = "Adv Example(ImgID {}) is not VALID (due to the original img inference error), has been abandoned.".format(adv_img_file_id)
            reporter.console_log(msg, Fore.GREEN, save_db=False, send_msg=False, show_batch=True, show_step_sequence=True)
            continue

        for adv_img_inference_log in adv_img_inference_log_list:
            for ori_img_inference_log in ori_img_inference_log_list:
                if adv_img_inference_log["inference_model"] == ori_img_inference_log["inference_model"]:
                    # 寻找两个记录中模型一致的(在同一个模型上进行的测试)

                    ori_inference_label = ori_img_inference_log["inference_img_label"] # 原始图片预测的标签
                    adv_inference_label = adv_img_inference_log["inference_img_label"] # 对抗图片预测的标签

                    if ori_inference_label == ori_label: # 原始图片在该测试模型上预测准确(不准确的直接无效处理)
                        inference_model = adv_img_inference_log["inference_model"]
                        # 初始化每个模型的测试结果统计
                        if attack_test_result.get(inference_model, None) is None:
                            attack_test_result[inference_model] = {"Mc": [], "IAC": [], "RTC": []}
                        # 获取误分类数量(Mc)
                        if adv_inference_label != ori_label:
                            attack_test_result[inference_model]["Mc"].append(1)
                        else:
                            attack_test_result[inference_model]["Mc"].append(0)
                        # 获取置信偏移
                        attack_test_result[inference_model]["IAC"] \
                            .append(adv_img_inference_log["inference_img_conf_array"][adv_inference_label] -
                                    ori_img_inference_log["inference_img_conf_array"][adv_inference_label])
                        attack_test_result[inference_model]["RTC"] \
                            .append(ori_img_inference_log["inference_img_conf_array"][ori_label] -
                                    adv_img_inference_log["inference_img_conf_array"][ori_label])

    for inference_model in attack_test_result:
        MR = calc_average(attack_test_result[inference_model]["Mc"])
        AIAC = calc_average(attack_test_result[inference_model]["IAC"])
        ARTC = calc_average(attack_test_result[inference_model]["RTC"])

        # 写入日志
        save_attack_deflection_capability_indicator_data(attack_info['atk_name'], attack_info['base_model'], inference_model, adv_example_file_type,
                                   MR, AIAC, ARTC, atk_perturbation_budget=attack_info['atk_perturbation_budget'])
    # 增加计数
    batch_manager.sys_log_logger.update_completed_num(1)
    batch_manager.sys_log_logger.update_finish_status(True)


def attack_adv_example_da_analyzer_and_evaluation(atk_name, base_model, use_raw_nparray_data=False):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}*)'s adv example disturbance-aware test result".format(atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model)
    if is_skip:
        return
    attack_info = find_attack_log_by_name_and_base_model(atk_name, base_model)

    attack_adv_example_da_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data)


def attack_adv_example_da_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data=False):
    all_adv_example_log = find_adv_example_file_logs_by_attack_id(attack_info['attack_id'])

    adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    analyzer_log = {
        "CT": [], "MD": [], "ED": [], "PCR": [], "DMS": [], "LMS": []
    }
    for adv_example_log in all_adv_example_log:
        # 获取消耗时间
        analyzer_log["CT"].append(adv_example_log["cost_time"])
        # 获取对抗样本扰动测试项目
        adv_example_da_test_data = find_adv_example_da_test_data_by_id_and_type(adv_example_log["adv_img_file_id"], adv_example_file_type)
        analyzer_log["MD"].append(float(adv_example_da_test_data["maximum_disturbance"]))
        analyzer_log["ED"].append(float(adv_example_da_test_data["euclidean_distortion"]))
        analyzer_log["PCR"].append(float(adv_example_da_test_data["pixel_change_ratio"]))
        analyzer_log["DMS"].append(float(adv_example_da_test_data["deep_metrics_similarity"]))
        analyzer_log["LMS"].append(float(adv_example_da_test_data["low_level_metrics_similarity"]))

    ACT = calc_average(analyzer_log["CT"])
    AMD = calc_average(analyzer_log["MD"])
    AED = calc_average(analyzer_log["ED"])
    APCR = calc_average(analyzer_log["PCR"])
    ADMS = calc_average(analyzer_log["DMS"])
    ALMS = calc_average(analyzer_log["LMS"])

    # 写入日志
    save_attack_adv_example_da_indicator_data(attack_info['atk_name'], attack_info['base_model'], adv_example_file_type,
                                              ACT, AMD, AED, APCR, ADMS, ALMS,
                                              atk_perturbation_budget=attack_info['atk_perturbation_budget'])
    # 增加计数
    batch_manager.sys_log_logger.update_completed_num(1)
    batch_manager.sys_log_logger.update_finish_status(True)


def attack_capability_with_perturbation_increment_analyzer_and_evaluation(atk_name, base_model, use_raw_nparray_data=False):
    msg = "统计攻击方法 {} (基于 {} 模型) 生成的对抗样本扰动探索结果".format(atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)
    participant = atk_name + ":" + base_model
    participant += "(RAW)" if use_raw_nparray_data else "(IMG)"
    is_skip, completed_num = global_recovery.check_skip(participant)
    if is_skip:
        return

    attack_logs = find_attack_log_by_name_and_base_model(atk_name, base_model, perturbation_increment_mode=True)
    for attack_info in attack_logs:
        attack_deflection_capability_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data)
        attack_adv_example_da_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data)
