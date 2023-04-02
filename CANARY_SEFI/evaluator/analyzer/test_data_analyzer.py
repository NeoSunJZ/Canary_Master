from colorama import Fore
from tqdm import tqdm

from CANARY_SEFI.core.function.basic.dataset.adv_dataset_function import adv_dataset_single_image_reader
from CANARY_SEFI.core.function.basic.dataset.dataset_function import dataset_single_image_reader
from CANARY_SEFI.core.function.basic.dataset.trans_dataset_function import trans_dataset_single_image_reader
from CANARY_SEFI.evaluator.logger.trans_file_info_handler import find_adv_trans_file_logs_by_attack_id_and_trans_name, \
    set_adv_trans_file_ground_valid
from CANARY_SEFI.handler.image_handler.img_utils import get_img_cosine_similarity, img_size_uniform_fix
from CANARY_SEFI.handler.image_handler.plt_handler import cam_diff_fig_builder, figure_show_handler
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_da_test_data_handler import find_adv_example_da_test_data_by_id_and_type
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id, \
    set_adv_example_file_ground_valid, find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.evaluator.logger.indicator_data_handler import save_attack_deflection_capability_indicator_data, \
    save_attack_adv_example_da_indicator_data, save_model_capability_indicator_data, \
    save_attack_adv_example_cost_indicator_data, save_trans_deflection_capability_indicator_data
from CANARY_SEFI.evaluator.logger.attack_info_handler import find_attack_log_by_name_and_base_model
from sklearn.metrics import f1_score, accuracy_score

from CANARY_SEFI.evaluator.logger.inference_test_data_handler import get_inference_test_data_by_model_name, \
    get_inference_test_data_by_img_id
from CANARY_SEFI.evaluator.analyzer.analyzer_tools import calc_average


def model_inference_capability_analyzer_and_evaluation(model_name):
    msg = "Analyzing and Evaluating Model({})'s inference capability test result".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

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
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)


def attack_deflection_capability_analyzer_and_evaluation(atk_name, base_model, dataset_info=None,
                                                         use_raw_nparray_data=False):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}*)'s deflection capability test result".format(atk_name,
                                                                                                          base_model)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model)
    if is_skip:
        return
    attack_info = find_attack_log_by_name_and_base_model(atk_name, base_model)
    attack_deflection_capability_analyzer_and_evaluation_handler(attack_info, dataset_info, use_raw_nparray_data)


def attack_deflection_capability_analyzer_and_evaluation_handler(attack_info, dataset_info=None,
                                                                 use_raw_nparray_data=False):
    all_adv_example_file_log = find_adv_example_file_logs_by_attack_id(attack_info['attack_id'])

    attack_test_result = {}

    adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    for adv_example_file_log in tqdm(all_adv_example_file_log, desc="Adv-example Inference Result Analyze Progress"):
        ori_img_id = adv_example_file_log["ori_img_id"]  # 原始图片ID(干净样本O)
        adv_img_file_id = adv_example_file_log["adv_img_file_id"]  # 对抗样本ID(干净样本O通过在M1上的攻击产生对抗样本A)

        ori_img_log = find_img_log_by_id(ori_img_id)  # 原始图片记录
        ori_label = ori_img_log["ori_img_label"]  # 原始图片的标签
        adv_target_label = adv_example_file_log["tlabel"]  # 定向攻击的目标label

        # 原始图片的预测记录(如干净样本O在M1、M2、M3、M4上进行了测试则有四条)
        ori_img_inference_log_list = get_inference_test_data_by_img_id(ori_img_id, DatasetType.NORMAL.value)
        # 确定原始样本有效性(目标模型必须是预测准确的，否则原始样本和其生成的对抗样本都无效)
        is_valid = True
        for ori_img_inference_log in ori_img_inference_log_list:
            # 干净样本O通过在M1上的攻击产生对抗样本A，这里需要检查M1能否准确推理干净样本O的标签
            if ori_img_inference_log["inference_model"] == attack_info['base_model']:
                is_valid = ori_img_inference_log["inference_img_label"] == ori_label
        if not is_valid:
            msg = "Adv Example(ImgID {}) is not VALID (due to the original img inference error (ON GENERATE MODEL)), has been abandoned.".format(
                adv_img_file_id)
            reporter.console_log(msg, Fore.GREEN, save_db=False, send_msg=False, show_task=True,
                                 show_step_sequence=True)

            continue

        # 对抗样本的预测记录(如对抗样本A在M1、M2、M3、M4上进行了测试则有四条)
        adv_img_inference_log_list = get_inference_test_data_by_img_id(adv_img_file_id, adv_example_file_type)
        with tqdm(total=len(adv_img_inference_log_list),
                  desc="Adv-example Inference Result Analyze Progress (Per Inference Model)", ncols=120) as bar:
            # 遍历对抗样本推理记录
            for adv_img_inference_log in adv_img_inference_log_list:
                # 遍历干净样本推理记录
                for ori_img_inference_log in ori_img_inference_log_list:
                    # 寻找两个记录中模型一致的(在同一个模型上进行的测试)，不一致的跳过即可
                    if adv_img_inference_log["inference_model"] != ori_img_inference_log["inference_model"]:
                        continue

                    # A在M1的预测对应O在M1的预测\A在M2的预测对应O在M2的预测\以此类推
                    # 由于A是由干净样本O通过在M1上的攻击产生，A在非M1的预测都是转移测试
                    ori_inference_label = ori_img_inference_log["inference_img_label"]  # 原始图片预测的标签
                    adv_inference_label = adv_img_inference_log["inference_img_label"]  # 对抗图片预测的标签

                    if adv_img_inference_log["inference_model"] != attack_info['base_model']:
                        if ori_inference_label != ori_label:
                            # 已经判断在目标模型上的准确性，此处无需再判断
                            # 原始图片必须在测试模型(转移模型)上也预测准确(不准确的直接无效处理)
                            msg = "Adv Example(ImgID {}) is not VALID (due to the original img inference error (ON TEST MODEL)), has been temporarily abandoned.".format(
                                adv_img_file_id)
                            reporter.console_log(msg, Fore.GREEN, save_db=False, send_msg=False, show_task=True,
                                                 show_step_sequence=True)
                            bar.update(1)
                            continue
                        test_on_base_model = False
                    else:
                        test_on_base_model = True

                    inference_model = adv_img_inference_log["inference_model"]
                    # 初始化每个模型的测试结果统计
                    if attack_test_result.get(inference_model, None) is None:
                        attack_test_result[inference_model] = {"Mc": [], "TAS": [], "IAC": [], "RTC": [], "CAMC_A": [], "CAMC_T": []}
                    # 获取误分类数量(Mc:Misclassification)
                    if adv_inference_label != ori_label:  # 攻击是否成功
                        attack_test_result[inference_model]["Mc"].append(1)
                        success_flag = True
                    else:
                        attack_test_result[inference_model]["Mc"].append(0)
                        success_flag = False

                    # 定向攻击成功率(TAS:Targeted Attack Success)
                    if adv_target_label != "None" and adv_target_label is not None:
                        if str(adv_inference_label) == str(adv_target_label):
                            attack_test_result[inference_model]["TAS"].append(1)
                        else:
                            attack_test_result[inference_model]["TAS"].append(0)

                    # 如果对抗样本没有设置有效性，且当前处理的是目标模型（而非迁移模型），则为其设置有效性
                    if adv_img_inference_log["inference_model"] == attack_info['base_model'] and \
                            adv_example_file_log["ground_valid"] is None:
                        set_adv_example_file_ground_valid(adv_img_file_id, success_flag)

                    # 获取置信偏移(IAC:Increase adversarial-class confidence/RTC:Reduction true-class confidence)
                    attack_test_result[inference_model]["IAC"] \
                        .append(adv_img_inference_log["inference_img_conf_array"][adv_inference_label] -
                                ori_img_inference_log["inference_img_conf_array"][adv_inference_label])
                    attack_test_result[inference_model]["RTC"] \
                        .append(ori_img_inference_log["inference_img_conf_array"][ori_label] -
                                adv_img_inference_log["inference_img_conf_array"][ori_label])
                    # 获取注意力偏移(CAMC_A:G-CAM Change(Adversarial-class)/CAMC_T: G-CAM Change(True-class))
                    CAMC_A = get_img_cosine_similarity(adv_img_inference_log["inference_class_cams"],
                                                       ori_img_inference_log["inference_class_cams"])
                    CAMC_T = get_img_cosine_similarity(adv_img_inference_log["true_class_cams"],
                                                       ori_img_inference_log["true_class_cams"])
                    # CAM感知为空时，不能计算余弦相似度，否则会造成严重问题
                    if CAMC_T is not None and CAMC_A is not None:
                        attack_test_result[inference_model]["CAMC_A"].append(CAMC_A)
                        attack_test_result[inference_model]["CAMC_T"].append(CAMC_T)

                    # 执行CAM可解释性偏移对比分析
                    cam_adv = adv_img_inference_log.get("true_class_cams")
                    cam_ori = ori_img_inference_log.get("true_class_cams")
                    true_class_cams = (cam_ori, cam_adv)
                    cam_adv = adv_img_inference_log.get("inference_class_cams")
                    cam_ori = ori_img_inference_log.get("inference_class_cams")
                    inference_class_cams = (cam_ori, cam_adv)
                    # (必须先有推理结果才可在此测试， 否则跳过)
                    if true_class_cams[0] is None or true_class_cams[1] is None or inference_class_cams[0] is None or inference_class_cams[1] is None:
                        continue
                    if dataset_info is not None:
                        ori_img, _ = dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])
                        adv_img = adv_dataset_single_image_reader(adv_example_file_log,
                                                                  DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA
                                                                  if use_raw_nparray_data else
                                                                  DatasetType.ADVERSARIAL_EXAMPLE_IMG)
                    else:
                        ori_img, adv_img = None, None

                    atk_name = "{}({})".format(attack_info.get("atk_name"), attack_info.get("base_model"))
                    if attack_info.get("atk_perturbation_budget") != 'None' and attack_info.get("atk_perturbation_budget") is not None:
                        atk_name + ":(e-{})".format(str(round(float(attack_info.get("atk_perturbation_budget")), 5)))

                    model_name = ori_img_inference_log["inference_model"]
                    ori_img, adv_img = img_size_uniform_fix(ori_img, adv_img, use_raw_nparray_data)
                    cam_result_plt = cam_diff_fig_builder((ori_img, adv_img), true_class_cams, inference_class_cams,
                                                          info=(
                                                              model_name, atk_name, ori_img_id, adv_img_file_id, ori_label,
                                                              ori_inference_label, adv_inference_label
                                                          ))
                    figure_show_handler(cam_result_plt,
                                        file_path="Grad-CAM_analyze_result/",
                                        file_name="adv_{}(Attack{})(InferenceModel{})".format(adv_img_file_id, atk_name, model_name))
                bar.update(1)

    for inference_model in attack_test_result:
        MR = calc_average(attack_test_result[inference_model]["Mc"])
        # 仅定向攻击使用该测评量
        if len(attack_test_result[inference_model]["TAS"])>0:
            # TASR: Targeted Attack Success Rate
            TASR = calc_average(attack_test_result[inference_model]["TAS"])
            MR = "{}/{}".format(MR, TASR)  # 存储时和MR存储在一起
        AIAC = calc_average(attack_test_result[inference_model]["IAC"])
        ARTC = calc_average(attack_test_result[inference_model]["RTC"])
        ACAMC_A = calc_average(attack_test_result[inference_model]["CAMC_A"])
        ACAMC_T = calc_average(attack_test_result[inference_model]["CAMC_T"])

        # 写入日志
        save_attack_deflection_capability_indicator_data(attack_info['atk_name'], attack_info['base_model'],
                                                         inference_model, adv_example_file_type,
                                                         MR, AIAC, ARTC, ACAMC_A, ACAMC_T,
                                                         atk_perturbation_budget=attack_info['atk_perturbation_budget'])
    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)


def attack_adv_example_da_and_cost_analyzer_and_evaluation(atk_name, base_model, use_raw_nparray_data=False):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}*)'s adv example disturbance-aware test result".format(
        atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model)
    if is_skip:
        return
    attack_info = find_attack_log_by_name_and_base_model(atk_name, base_model)

    attack_adv_example_da_and_cost_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data)


def attack_adv_example_da_and_cost_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data=False):
    all_adv_example_file_log = find_adv_example_file_logs_by_attack_id(attack_info['attack_id'])

    adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    analyzer_log = {
        "CT": [], "QN_F": [], "QN_B": [], "MD": [], "ED": [], "ED_HF": [], "ED_LF": [], "PCR": [], "DMS": [], "LMS": []
    }
    for adv_example_file_log in all_adv_example_file_log:
        ground_valid = adv_example_file_log["ground_valid"]
        if ground_valid != "1":  # 对抗样本在目标模型上攻击失败的不计入以避免误差
            continue
        # 获取消耗时间
        analyzer_log["CT"].append(adv_example_file_log["cost_time"])
        # 获取查询次数
        # 老版本数据无此项目，需要None改成0
        adv_example_file_log["query_num_forward"] = 0 if adv_example_file_log["query_num_forward"] is None else adv_example_file_log["query_num_forward"]
        adv_example_file_log["query_num_backward"] = 0 if adv_example_file_log["query_num_backward"] is None else adv_example_file_log["query_num_backward"]
        analyzer_log["QN_F"].append(adv_example_file_log["query_num_forward"])
        analyzer_log["QN_B"].append(adv_example_file_log["query_num_backward"])

        # 获取对抗样本扰动测试项目
        adv_example_da_test_data = find_adv_example_da_test_data_by_id_and_type(adv_example_file_log["adv_img_file_id"],
                                                                                adv_example_file_type)
        analyzer_log["MD"].append(float(adv_example_da_test_data["maximum_disturbance"]))
        analyzer_log["ED"].append(float(adv_example_da_test_data["euclidean_distortion"]))
        analyzer_log["ED_HF"].append(float(adv_example_da_test_data["high_freq_euclidean_distortion"]))
        analyzer_log["ED_LF"].append(float(adv_example_da_test_data["low_freq_euclidean_distortion"]))
        analyzer_log["PCR"].append(float(adv_example_da_test_data["pixel_change_ratio"]))
        analyzer_log["DMS"].append(float(adv_example_da_test_data["deep_metrics_similarity"]))
        analyzer_log["LMS"].append(float(adv_example_da_test_data["low_level_metrics_similarity"]))
    # COST
    ACT = calc_average(analyzer_log["CT"])
    AQN_F = calc_average(analyzer_log["QN_F"])
    AQN_B = calc_average(analyzer_log["QN_B"])
    # DA
    AMD = calc_average(analyzer_log["MD"])
    AED = calc_average(analyzer_log["ED"])
    AED_HF = calc_average(analyzer_log["ED_HF"])
    AED_LF = calc_average(analyzer_log["ED_LF"])
    APCR = calc_average(analyzer_log["PCR"])
    ADMS = calc_average(analyzer_log["DMS"])
    ALMS = calc_average(analyzer_log["LMS"])

    # 写入日志
    # COST日志
    save_attack_adv_example_cost_indicator_data(attack_info['atk_name'], attack_info['base_model'],
                                                ACT, AQN_F, AQN_B,
                                                atk_perturbation_budget=attack_info['atk_perturbation_budget'])
    # DA日志
    save_attack_adv_example_da_indicator_data(attack_info['atk_name'], attack_info['base_model'], adv_example_file_type,
                                              AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS,
                                              atk_perturbation_budget=attack_info['atk_perturbation_budget'])

    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)


def attack_capability_with_perturbation_increment_analyzer_and_evaluation(atk_name, base_model, dataset_info=None,
                                                                          use_raw_nparray_data=False):
    msg = "统计攻击方法 {} (基于 {} 模型) 生成的对抗样本扰动探索结果".format(atk_name, base_model)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
    participant = atk_name + ":" + base_model
    participant += "(RAW)" if use_raw_nparray_data else "(IMG)"
    is_skip, completed_num = global_recovery.check_skip(participant)
    if is_skip:
        return

    attack_logs = find_attack_log_by_name_and_base_model(atk_name, base_model, perturbation_increment_mode=True)
    for attack_info in attack_logs:
        attack_deflection_capability_analyzer_and_evaluation_handler(attack_info, dataset_info, use_raw_nparray_data)
        attack_adv_example_da_and_cost_analyzer_and_evaluation_handler(attack_info, use_raw_nparray_data)


def trans_deflection_capability_analyzer_and_evaluation(atk_name, base_model, trans_name, dataset_info=None,
                                                        use_raw_nparray_data=False):
    msg = "Analyzing and Evaluating Method({} *BaseModel {}* Trans Method {})'s deflection capability test result"\
        .format(atk_name, base_model, trans_name)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(atk_name + ":" + base_model + ":" + trans_name)
    if is_skip:
        return
    attack_info = find_attack_log_by_name_and_base_model(atk_name, base_model)
    trans_logs = find_adv_trans_file_logs_by_attack_id_and_trans_name(attack_info['attack_id'], trans_name)
    trans_deflection_capability_analyzer_and_evaluation_handler(attack_info, trans_logs, trans_name, dataset_info, use_raw_nparray_data)


def trans_deflection_capability_analyzer_and_evaluation_handler(attack_info, trans_logs, trans_name, dataset_info=None,
                                                                use_raw_nparray_data=False):

    attack_test_result = {}

    trans_file_type = DatasetType.TRANSFORM_RAW_DATA.value if use_raw_nparray_data else DatasetType.TRANSFORM_IMG.value

    for trans_log in tqdm(trans_logs, desc="Trans-example Inference Result Analyze Progress"):
        adv_example_file_log = find_adv_example_file_log_by_id(trans_log['adv_img_file_id'])
        ori_img_id = adv_example_file_log["ori_img_id"]  # 原始图片ID(干净样本O)
        adv_img_file_id = adv_example_file_log["adv_img_file_id"]  # 对抗样本ID(干净样本O通过在M1上的攻击产生对抗样本A)

        ori_img_log = find_img_log_by_id(ori_img_id)  # 原始图片记录
        ori_label = ori_img_log["ori_img_label"]  # 原始图片的标签
        adv_target_label = adv_example_file_log["tlabel"]  # 定向攻击的目标label

        # 原始图片的预测记录(如干净样本O在M1、M2、M3、M4上进行了测试则有四条)
        ori_img_inference_log_list = get_inference_test_data_by_img_id(ori_img_id, DatasetType.NORMAL.value)
        # 确定原始样本有效性(目标模型必须是预测准确的，否则原始样本和其生成的对抗样本都无效)
        is_valid = True
        for ori_img_inference_log in ori_img_inference_log_list:
            # 干净样本O通过在M1上的攻击产生对抗样本A，这里需要检查M1能否准确推理干净样本O的标签
            if ori_img_inference_log["inference_model"] == attack_info['base_model']:
                is_valid = ori_img_inference_log["inference_img_label"] == ori_label
        if not is_valid:
            msg = "Trans Example(ImgID {}) is not VALID (due to the original img inference error (ON GENERATE MODEL)), " \
                  "has been abandoned.".format(adv_img_file_id)
            reporter.console_log(msg, Fore.GREEN, save_db=False, send_msg=False, show_task=True,
                                 show_step_sequence=True)

            continue

        # 防御样本的预测记录(如对抗样本A在M1、M2、M3、M4上进行了测试则有四条)
        trans_img_inference_log_list = get_inference_test_data_by_img_id(trans_log['adv_trans_img_file_id'], trans_file_type)

        with tqdm(total=len(trans_img_inference_log_list),
                  desc="Trans-example Inference Result Analyze Progress (Per Inference Model)", ncols=120) as bar:
            # 遍历防御样本推理记录
            for trans_img_inference_log in trans_img_inference_log_list:
                # 遍历干净样本推理记录
                for ori_img_inference_log in ori_img_inference_log_list:
                    # 寻找两个记录中模型一致的(在同一个模型上进行的测试)，不一致的跳过即可
                    if trans_img_inference_log["inference_model"] != ori_img_inference_log["inference_model"]:
                        continue

                    # A在M1的预测对应O在M1的预测\A在M2的预测对应O在M2的预测\以此类推
                    # 由于A是由干净样本O通过在M1上的攻击产生，A在非M1的预测都是转移测试
                    ori_inference_label = ori_img_inference_log["inference_img_label"]  # 原始图片预测的标签
                    trans_inference_label = trans_img_inference_log["inference_img_label"]  # 防御图片预测的标签

                    if trans_img_inference_log["inference_model"] != attack_info['base_model']:
                        if ori_inference_label != ori_label:
                            # 已经判断在目标模型上的准确性，此处无需再判断
                            # 原始图片必须在测试模型(转移模型)上也预测准确(不准确的直接无效处理)
                            msg = "Trans Example(ImgID {}) is not VALID (due to the original img inference error (ON TEST MODEL)), has been temporarily abandoned.".format(
                                trans_log['adv_trans_img_file_id'])
                            reporter.console_log(msg, Fore.GREEN, save_db=False, send_msg=False, show_task=True,
                                                 show_step_sequence=True)
                            bar.update(1)
                            continue
                        test_on_base_model = False
                    else:
                        test_on_base_model = True

                    inference_model = trans_img_inference_log["inference_model"]
                    # 初始化每个模型的测试结果统计
                    if attack_test_result.get(inference_model, None) is None:
                        attack_test_result[inference_model] = {"Mc": [], "TAS": [], "IAC": [], "RTC": [], "CAMC_A": [], "CAMC_T": []}
                    # 获取误分类数量(Mc:Misclassification)
                    if trans_inference_label != ori_label:  # 攻击是否成功
                        attack_test_result[inference_model]["Mc"].append(1)
                        success_flag = True
                    else:
                        attack_test_result[inference_model]["Mc"].append(0)
                        success_flag = False

                    # 定向攻击成功率(TAS:Targeted Attack Success)
                    if adv_target_label != "None" and adv_target_label is not None:
                        if str(trans_inference_label) == str(adv_target_label):
                            attack_test_result[inference_model]["TAS"].append(1)
                        else:
                            attack_test_result[inference_model]["TAS"].append(0)

                    # 如果对抗样本没有设置有效性，且当前处理的是目标模型（而非迁移模型），则为其设置有效性
                    if trans_img_inference_log["inference_model"] == attack_info['base_model'] and \
                            trans_log["ground_valid"] is None:
                        set_adv_trans_file_ground_valid(trans_log['adv_trans_img_file_id'], success_flag)

                    # 获取置信偏移(IAC:Increase adversarial-class confidence/RTC:Reduction true-class confidence)
                    attack_test_result[inference_model]["IAC"] \
                        .append(trans_img_inference_log["inference_img_conf_array"][trans_inference_label] -
                                ori_img_inference_log["inference_img_conf_array"][trans_inference_label])
                    attack_test_result[inference_model]["RTC"] \
                        .append(ori_img_inference_log["inference_img_conf_array"][ori_label] -
                                trans_img_inference_log["inference_img_conf_array"][ori_label])
                    # 获取注意力偏移(CAMC_A:G-CAM Change(Adversarial-class)/CAMC_T: G-CAM Change(True-class))
                    CAMC_A = get_img_cosine_similarity(trans_img_inference_log["inference_class_cams"],
                                                       ori_img_inference_log["inference_class_cams"])
                    CAMC_T = get_img_cosine_similarity(trans_img_inference_log["true_class_cams"],
                                                       ori_img_inference_log["true_class_cams"])
                    # CAM感知为空时，不能计算余弦相似度，否则会造成严重问题
                    if CAMC_T is not None and CAMC_A is not None:
                        attack_test_result[inference_model]["CAMC_A"].append(CAMC_A)
                        attack_test_result[inference_model]["CAMC_T"].append(CAMC_T)

                    # 执行CAM可解释性偏移对比分析(必须先有推理结果才可在此测试， 否则跳过)
                    cam_adv = trans_img_inference_log.get("true_class_cams")
                    cam_ori = ori_img_inference_log.get("true_class_cams")
                    true_class_cams = (cam_ori, cam_adv)
                    cam_adv = trans_img_inference_log.get("inference_class_cams")
                    cam_ori = ori_img_inference_log.get("inference_class_cams")
                    inference_class_cams = (cam_ori, cam_adv)
                    # (必须先有推理结果才可在此测试， 否则跳过)
                    if true_class_cams[0] is None or true_class_cams[1] is None or inference_class_cams[0] is None or \
                            inference_class_cams[1] is None:
                        continue
                    if dataset_info is not None:
                        ori_img, _ = dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])
                        trans_img = trans_dataset_single_image_reader(trans_log,
                                                                      DatasetType.TRANSFORM_RAW_DATA
                                                                      if use_raw_nparray_data else
                                                                      DatasetType.TRANSFORM_IMG)
                    else:
                        ori_img, trans_img = None, None

                    transform_name = "{}({}):(e-{})".format(attack_info.get("atk_name"), attack_info.get("base_model"), trans_name)

                    model_name = ori_img_inference_log["inference_model"]
                    ori_img, trans_img = img_size_uniform_fix(ori_img, trans_img)
                    cam_result_plt = cam_diff_fig_builder((ori_img, trans_img), true_class_cams, inference_class_cams,
                                                          info=(
                                                              model_name, attack_info['atk_name']+":"+trans_name,
                                                              ori_img_id, trans_log['adv_trans_img_file_id'], ori_label,
                                                              ori_inference_label, trans_inference_label
                                                          ))
                    figure_show_handler(cam_result_plt,
                                        file_path="Grad-CAM_analyze_result/"+trans_name+"/",
                                        file_name="trans_{}(Attack{})(InferenceModel{})(Trans{})"
                                        .format(trans_log['adv_trans_img_file_id'], attack_info['atk_name'],
                                                model_name, trans_name))
                bar.update(1)

    for inference_model in attack_test_result:
        MR = calc_average(attack_test_result[inference_model]["Mc"])
        # 仅定向攻击使用该测评量
        if len(attack_test_result[inference_model]["TAS"])>0:
            # TASR: Targeted Attack Success Rate
            TASR = calc_average(attack_test_result[inference_model]["TAS"])
            MR = "{}/{}".format(MR, TASR)  # 存储时和MR存储在一起
        AIAC = calc_average(attack_test_result[inference_model]["IAC"])
        ARTC = calc_average(attack_test_result[inference_model]["RTC"])
        ACAMC_A = calc_average(attack_test_result[inference_model]["CAMC_A"])
        ACAMC_T = calc_average(attack_test_result[inference_model]["CAMC_T"])

        # 写入日志
        save_trans_deflection_capability_indicator_data(attack_info['atk_name'], attack_info['base_model'], trans_name,
                                                        inference_model, trans_file_type,
                                                        MR, AIAC, ARTC, ACAMC_A, ACAMC_T,)
    # 增加计数
    task_manager.sys_log_logger.update_completed_num(1)
    task_manager.sys_log_logger.update_finish_status(True)
