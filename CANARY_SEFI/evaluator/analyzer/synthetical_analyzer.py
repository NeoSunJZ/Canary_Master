from colorama import Fore

from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.indicator_data_handler import get_model_capability_indicator_data, \
    get_attack_deflection_capability_indicator_data_by_base_model, \
    get_attack_adv_example_da_indicator_data_by_base_model, add_model_security_synthetical_capability_log
from CANARY_SEFI.handler.tools.analyzer_tools import calc_average


def model_security_synthetical_capability_analyzer_and_evaluation(model_name, use_raw_nparray_data=False):
    msg = "Analyze Model({})'s synthetical capability evaluation result.".format(model_name)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    is_skip, completed_num = global_recovery.check_skip(model_name)
    if is_skip:
        return
    model_capability_indicator_data = get_model_capability_indicator_data(model_name)
    adversarial_example_analyzer_log = {
        "MR": [], "AIAC": [], "ARTC": [], "ACT": [], "AMD": [], "AED": [], "APCR": [], "ADMS": [], "ALMS": []
    }
    attack_deflection_capability_indicator_data = get_attack_deflection_capability_indicator_data_by_base_model(model_name)
    attack_adv_example_da_indicator_data = get_attack_adv_example_da_indicator_data_by_base_model(model_name)

    for log in attack_deflection_capability_indicator_data:
        # 排除转移测试
        if log["base_model"] == log["inference_model"]:
            adversarial_example_analyzer_log['MR'].append(float(log["MR"]))
            adversarial_example_analyzer_log['AIAC'].append(float(log["AIAC"]))
            adversarial_example_analyzer_log['ARTC'].append(float(log["ARTC"]))
    for log in attack_adv_example_da_indicator_data:
        adversarial_example_analyzer_log['ACT'].append(float(log["ACT"]))
        adversarial_example_analyzer_log['AMD'].append(float(log["AMD"]))
        adversarial_example_analyzer_log['AED'].append(float(log["AED"]))
        adversarial_example_analyzer_log['APCR'].append(float(log["APCR"]))
        adversarial_example_analyzer_log['ADMS'].append(float(log["ADMS"]))
        adversarial_example_analyzer_log['ALMS'].append(float(log["ALMS"]))

    model_ACC = float(model_capability_indicator_data["clear_acc"])
    model_F1 = float(model_capability_indicator_data["clear_f1"])
    model_Conf = float(model_capability_indicator_data["clear_conf"])

    model_MR = calc_average(adversarial_example_analyzer_log["MR"])
    model_AIAC =calc_average(adversarial_example_analyzer_log["AIAC"])
    model_ARTC = calc_average(adversarial_example_analyzer_log["ARTC"])
    model_ACT = calc_average(adversarial_example_analyzer_log["ACT"])
    model_AMD = calc_average(adversarial_example_analyzer_log["AMD"])
    model_AED =calc_average(adversarial_example_analyzer_log["AED"])
    model_APCR = calc_average(adversarial_example_analyzer_log["APCR"])
    model_ADMS = calc_average(adversarial_example_analyzer_log["ADMS"])
    model_ALMS = calc_average(adversarial_example_analyzer_log["ALMS"])

    test_adv_example_file_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA.value if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG.value

    add_model_security_synthetical_capability_log(model_name, test_adv_example_file_type,
                                                  model_ACC, model_F1, model_Conf,
                                                  model_MR, model_AIAC, model_ARTC, model_ACT,
                                                  model_AMD, model_AED, model_APCR, model_ADMS, model_ALMS)
