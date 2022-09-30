from colorama import Fore

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter

logger = batch_manager.test_data_logger


def save_model_capability_indicator_data(model_name, clear_acc, clear_f1, clear_conf):
    sql = "REPLACE INTO model_capability_indicator_data (model_name, clear_acc, clear_f1, clear_conf) " \
          "VALUES (?,?,?,?,?)"
    args = (str(model_name), clear_acc, clear_f1, clear_conf)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Model Indicator of Model({}) is [ ACC:{} F1:{} True-Class Conf:{} ].".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_model_capability_indicator_data(model_name):
    sql = "SELECT * FROM model_capability_indicator_data WHERE model_name = ?"
    return logger.query_log(sql, (model_name,))


# 缩写释义:
# MR: misclassification ratio
# AIAC: average increase adversarial-class confidence
# ARTC: average reduction true-class confidence
# ACT: average cost time
def save_attack_capability_indicator_data(atk_name, base_model, inference_model, adv_example_file_type,
                                          MR, AIAC, ARTC, ACT, atk_perturbation_budget=None):

    sql = "REPLACE INTO attack_capability_indicator_data (atk_name, base_model, atk_perturbation_budget, " \
          "inference_model, adv_example_file_type, MR, AIAC, ARTC, ACT) VALUES (?,?,?,?,?,?,?,?,?)"
    args = (str(atk_name), str(base_model), atk_perturbation_budget, str(inference_model), str(adv_example_file_type),
            MR, AIAC, ARTC, ACT)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Tested Adv example is generated by Attack Method({} *BaseModel {}* *PerturbationBudget {}*). " \
              "Tested Adv example is inferenced by Model({}). Test adv example file type:{}. " \
              "Attack Method Capability Indicator is [ MR:{} AIAC:{} ARTC:{} ACT:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_attack_capability_indicator_data_by_base_model(base_model):
    sql = " SELECT * FROM attack_capability_indicator_data WHERE base_model = ? "
    return logger.query_logs(sql, (base_model,))


def get_attack_capability_indicator_data_by_attack_name(atk_name):
    sql = " SELECT * FROM attack_capability_indicator_data WHERE atk_name = ? "
    return logger.query_logs(sql, (atk_name,))


# 缩写释义:
# AMD: average maximum disturbance
# AED: average euclidean distortion
# APCR: average pixel change ratio
# ADMS: average deep metrics similarity
# ALMS: average low-level metrics similarity
def save_attack_adv_example_da_indicator_data(atk_name, base_model, adv_example_file_type,
                                      AMD, AED, APCR, ADMS, ALMS, atk_perturbation_budget=None):
    sql = "INSERT INTO attack_adv_example_da_indicator_data " \
          "(atk_name, base_model, atk_perturbation_budget, adv_example_file_type, AMD, AED, APCR, ADMS, ALMS) " \
          "VALUES (?,?,?,?,?,?,?,?,?)"
    args = (str(atk_name), str(base_model), atk_perturbation_budget, str(adv_example_file_type), AMD, AED, APCR, ADMS, ALMS)
    logger.insert_log(sql, args)

    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Tested Adv example is generated by Attack Method({} *BaseModel {}* *PerturbationBudget {}*). " \
              "Test adv example file type:{}. " \
              "Attack Method DA Indicator is [ AMD:{} AED:{} APCR:{} ADMS:{} ALMS:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_attack_adv_example_da_indicator_data_by_base_model(base_model):
    sql = "SELECT * FROM attack_adv_example_da_indicator_data WHERE base_model = ? "
    return logger.query_logs(sql, (base_model,))


def get_attack_adv_example_da_indicator_data_by_attack_name(atk_name):
    sql = "SELECT * FROM attack_adv_example_da_indicator_data WHERE atk_name = ? "
    return logger.query_logs(sql, (atk_name,))


def get_explore_perturbation_result_by_attack_name_and_base_model(atk_name, base_model):
    sql = "SELECT * FROM attack_capability_indicator_data LEFT OUTER JOIN attack_adv_example_da_indicator_data ON " \
          "attack_capability_indicator_data.atk_name = attack_adv_example_da_indicator_data.atk_name " \
          "AND attack_capability_indicator_data.base_model = attack_adv_example_da_indicator_data.base_model " \
          "AND attack_capability_indicator_data.atk_perturbation_budget = attack_adv_example_da_indicator_data.atk_perturbation_budget " \
          "AND attack_capability_indicator_data.adv_example_file_type = attack_adv_example_da_indicator_data.adv_example_file_type " \
          "WHERE attack_capability_indicator_data.atk_name = ? AND attack_capability_indicator_data.base_model = ? " \
          "AND attack_capability_indicator_data.base_model = attack_capability_indicator_data.inference_model"
    return logger.query_logs(sql, (atk_name, base_model))


def add_model_security_synthetical_capability_log(model_name, model_ACC, model_F1, model_Conf,
                                                  model_MR, model_AIAC, model_ARTC, model_ACT,
                                                  model_AMD, model_AED, model_APCR, model_ADMS, model_ALMS):
    sql = "INSERT INTO model_dimension_summary (model_name, test_adv_example_file_type" \
          "ACC, F1, Conf, MR, AIAC, ARTC, ACT, AMD, AED, APCR, ADMS, ALMS) " \
          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
    args = (model_name, model_ACC, model_F1, model_Conf, model_MR, model_AIAC, model_ARTC, model_ACT,
            model_AMD, model_AED, model_APCR, model_ADMS, model_ALMS)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Model({})'s Evaluated Result (Tested Adv Example Type:{} ) :[ ACC:{} " \
              "F1:{} Conf:{} MR:{} AIAC:{} ARTC:{} ACT:{} AMD:{} AED:{} APCR:{} ADMS:{} ALMS:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_model_security_synthetical_capability_log(model_name):
    sql = " SELECT * FROM model_dimension_summary WHERE model_name = ? "
    return logger.query_log(sql, (model_name,))