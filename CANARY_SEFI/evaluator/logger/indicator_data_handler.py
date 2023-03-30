from colorama import Fore
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


def save_defense_normal_effectiveness_data(model_name, CAV, RRSR, ConV, COS, logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO defense_model_normal_inference_capability_indicator_data (model_name, CAV, RRSR, ConV, COS) " \
          "VALUES (?,?,?,?,?)"
    args = (str(model_name), CAV, RRSR, ConV, COS)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Model Indicator of Model({}) is [ CAV:{} RRSR:{} ConV:{} COS:{}].".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return

def save_defense_adv_effectiveness_data(model_name, attack_name, DCAV, TCAV, logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO defense_model_adv_inference_capability_indicator_data (model_name, attack_name, DCAV, TCAV) " \
          "VALUES (?,?,?,?)"
    args = (str(model_name), str(attack_name), DCAV, TCAV)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Model Indicator of Model({}) is [ DCAV:{} TCAV:{}].".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return

def save_model_capability_indicator_data(model_name, clear_acc, clear_f1, clear_conf, logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO model_inference_capability_indicator_data (model_name, clear_acc, clear_f1, clear_conf) " \
          "VALUES (?,?,?,?)"
    args = (str(model_name), clear_acc, clear_f1, clear_conf)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Model Indicator of Model({}) is [ ACC:{} F1:{} True-Class Conf:{} ].".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_model_capability_indicator_data(model_name):
    sql = "SELECT * FROM model_inference_capability_indicator_data WHERE model_name = ?"
    return task_manager.test_data_logger.query_log(sql, (model_name,))


# 缩写释义:
# MR: misclassification ratio
# AIAC: average increase adversarial-class confidence
# ARTC: average reduction true-class confidence
# ACAMC_A: average G-CAM Change (Adversarial-class)
# ACAMC_T: average G-CAM Change (True-class)
def save_attack_deflection_capability_indicator_data(atk_name, base_model, inference_model, adv_example_file_type,
                                                     MR, AIAC, ARTC, ACAMC_A, ACAMC_T, atk_perturbation_budget=None,
                                                     logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO attack_deflection_capability_indicator_data (atk_name, base_model, atk_perturbation_budget, " \
          "inference_model, adv_example_file_type, MR, AIAC, ARTC, ACAMC_A, ACAMC_T) VALUES (?,?,?,?,?,?,?,?,?,?)"
    args = (str(atk_name), str(base_model), atk_perturbation_budget, str(inference_model), str(adv_example_file_type),
            MR, AIAC, ARTC, ACAMC_A, ACAMC_T)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Tested Adv example is generated by Attack Method({} *BaseModel {}* *PerturbationBudget {}*). " \
              "Tested Adv example is inferenced by Model({}). Test adv example file type:{}. " \
              "Attack Method Capability Indicator is [ MR:{} AIAC:{} ARTC:{} ACAMC_A:{} ACAMC_T:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def save_trans_deflection_capability_indicator_data(atk_name, base_model, trans_name, inference_model, trans_file_type,
                                                    MR, AIAC, ARTC, ACAMC_A, ACAMC_T, logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO trans_deflection_capability_indicator_data (atk_name, base_model, trans_name, " \
          "inference_model, trans_file_type, MR, AIAC, ARTC, ACAMC_A, ACAMC_T) VALUES (?,?,?,?,?,?,?,?,?,?)"
    args = (str(atk_name), str(base_model), trans_name, str(inference_model), str(trans_file_type),
            MR, AIAC, ARTC, ACAMC_A, ACAMC_T)
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Tested Trans File is generated by Attack Method({} *BaseModel {}* *Trans name {}*). " \
              "Tested Trans file is inferenced by Model({}). Test trans file type:{}. " \
              "Trans Method Capability Indicator is [ MR:{} AIAC:{} ARTC:{} ACAMC_A:{} ACAMC_T:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_attack_deflection_capability_indicator_data_by_base_model(base_model):
    sql = " SELECT * FROM attack_deflection_capability_indicator_data WHERE base_model = ? "
    return task_manager.test_data_logger.query_logs(sql, (base_model,))


def get_attack_deflection_capability_indicator_data_by_attack_name(atk_name):
    sql = " SELECT * FROM attack_deflection_capability_indicator_data WHERE atk_name = ? "
    return task_manager.test_data_logger.query_logs(sql, (atk_name,))


# 缩写释义:
# AMD: average maximum disturbance
# AED: average euclidean distortion
# APCR: average pixel change ratio
# ADMS: average deep metrics similarity
# ALMS: average low-level metrics similarity
def save_attack_adv_example_da_indicator_data(atk_name, base_model, adv_example_file_type,
                                              AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS, atk_perturbation_budget=None,
                                              logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO attack_adv_example_da_indicator_data " \
          "(atk_name, base_model, atk_perturbation_budget, adv_example_file_type, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS) " \
          "VALUES (?,?,?,?,?,?,?,?,?,?,?)"
    args = (str(atk_name), str(base_model), atk_perturbation_budget, str(adv_example_file_type), AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS)
    logger.insert_log(sql, args)

    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Tested Adv example is generated by Attack Method({} *BaseModel {}* *PerturbationBudget {}*). " \
              "Test adv example file type:{}. " \
              "Attack Method DA Indicator is [ AMD:{} AED:{} AED_HF:{} AED_LF:{} APCR:{} ADMS:{} ALMS:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_attack_adv_example_da_indicator_data_by_base_model(base_model):
    sql = "SELECT * FROM attack_adv_example_da_indicator_data WHERE base_model = ? "
    return task_manager.test_data_logger.query_logs(sql, (base_model,))


def get_attack_adv_example_da_indicator_data_by_attack_name(atk_name):
    sql = "SELECT * FROM attack_adv_example_da_indicator_data WHERE atk_name = ? "
    return task_manager.test_data_logger.query_logs(sql, (atk_name,))



# 缩写释义:
# ACT: average cost time
# AQN_F: average query number (forward)
# AQN_B = average query number (backward)
def save_attack_adv_example_cost_indicator_data(atk_name, base_model, ACT, AQN_F, AQN_B, atk_perturbation_budget=None,
                                                logger=None):
    if logger is None:
        logger = task_manager.test_data_logger
    sql = "REPLACE INTO attack_adv_example_cost_indicator_data " \
          "(atk_name, base_model, atk_perturbation_budget, ACT, AQN_F, AQN_B) " \
          "VALUES (?,?,?,?,?,?)"
    args = (str(atk_name), str(base_model), atk_perturbation_budget, ACT, AQN_F, AQN_B)
    logger.insert_log(sql, args)

    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Tested Adv example is generated by Attack Method({} *BaseModel {}* *PerturbationBudget {}*). " \
              "Attack Method Cost Indicator is [ ACT:{} AQN_F:{} AQN_B:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_attack_adv_example_cost_indicator_data_by_base_model(base_model):
    sql = "SELECT * FROM attack_adv_example_cost_indicator_data WHERE base_model = ? "
    return task_manager.test_data_logger.query_logs(sql, (base_model,))


def get_attack_adv_example_cost_indicator_data_by_attack_name(atk_name):
    sql = "SELECT * FROM attack_adv_example_cost_indicator_data WHERE atk_name = ? "
    return task_manager.test_data_logger.query_logs(sql, (atk_name,))


def get_all_attack_deflection_capability_indicator_data():
    sql = "SELECT * FROM attack_deflection_capability_indicator_data"
    return task_manager.test_data_logger.query_logs(sql, ())


def get_all_attack_adv_example_da_indicator_data():
    sql = "SELECT * FROM attack_adv_example_da_indicator_data"
    return task_manager.test_data_logger.query_logs(sql, ())


def get_all_attack_adv_example_cost_indicator_data():
    sql = "SELECT * FROM attack_adv_example_cost_indicator_data"
    return task_manager.test_data_logger.query_logs(sql, ())



def get_attack_capability_with_perturbation_increment_indicator_data(atk_name, base_model):
    sql = "SELECT * FROM attack_deflection_capability_indicator_data LEFT OUTER JOIN attack_adv_example_da_indicator_data ON " \
          "attack_deflection_capability_indicator_data.atk_name = attack_adv_example_da_indicator_data.atk_name " \
          "AND attack_deflection_capability_indicator_data.base_model = attack_adv_example_da_indicator_data.base_model " \
          "AND attack_deflection_capability_indicator_data.atk_perturbation_budget = attack_adv_example_da_indicator_data.atk_perturbation_budget " \
          "AND attack_deflection_capability_indicator_data.adv_example_file_type = attack_adv_example_da_indicator_data.adv_example_file_type " \
          "WHERE attack_deflection_capability_indicator_data.atk_name = ? AND attack_deflection_capability_indicator_data.base_model = ? " \
          "AND attack_deflection_capability_indicator_data.base_model = attack_deflection_capability_indicator_data.inference_model"
    return task_manager.test_data_logger.query_logs(sql, (atk_name, base_model))


def add_model_security_synthetical_capability_log(model_name, test_adv_example_file_type,
                                                  model_ACC, model_F1, model_Conf,
                                                  MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T,
                                                  ACT, AQN_F, AQN_B,
                                                  AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS):
    sql = "INSERT INTO model_dimension_summary (model_name, test_adv_example_file_type, " \
          "ACC, F1, Conf, MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS) " \
          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    args = (model_name, test_adv_example_file_type, model_ACC, model_F1, model_Conf,
            MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS)
    task_manager.test_data_logger.insert_log(sql, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Model({})'s Evaluated Result (Tested Adv Example Type:{} ) :[ " \
              "ACC:{} F1:{} Conf:{} MR:{} TAS:{} AIAC:{} ARTC:{} ACAMC_A:{} ACAMC_T:{}" \
              "ACT:{} AQN_F:{} AQN_B:{} AMD:{} AED:{} AED_HF:{} AED_LF:{} APCR:{} ADMS:{} ALMS:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return


def get_model_security_synthetical_capability_log(model_name):
    sql = " SELECT * FROM model_dimension_summary WHERE model_name = ? "
    return task_manager.test_data_logger.query_log(sql, (model_name,))


def add_attack_synthetical_capability_log(attack_name, test_adv_example_file_type,
                                                  model_ACC, model_F1, model_Conf,
                                                  MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T,
                                                  ACT, AQN_F, AQN_B,
                                                  AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS):
    sql = "INSERT INTO attack_dimension_summary (attack_name, test_adv_example_file_type, " \
          "ACC, F1, Conf, MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS) " \
          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    args = (attack_name, test_adv_example_file_type, model_ACC, model_F1, model_Conf,
            MR, TAS, AIAC, ARTC, ACAMC_A, ACAMC_T, ACT, AQN_F, AQN_B, AMD, AED, AED_HF, AED_LF, APCR, ADMS, ALMS)
    task_manager.test_data_logger.insert_log(sql, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Model({})'s Evaluated Result (Tested Adv Example Type:{} ) :[ " \
              "ACC:{} F1:{} Conf:{} MR:{} TAS:{} AIAC:{} ARTC:{} ACAMC_A:{} ACAMC_T:{}" \
              "ACT:{} AQN_F:{} AQN_B:{} AMD:{} AED:{} AED_HF:{} AED_LF:{} APCR:{} ADMS:{} ALMS:{} ]".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return
