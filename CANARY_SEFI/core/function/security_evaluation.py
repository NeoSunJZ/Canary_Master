from colorama import Fore

from CANARY_SEFI.core.function.enum.step_enum import Step
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.core.function.inference import inference, adv_inference
from CANARY_SEFI.core.function.make_adv_example import build_AEs, explore_perturbation
from CANARY_SEFI.core.function.helper.batch_list_iterator import BatchListIterator
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo, DatasetType
from CANARY_SEFI.evaluator.analyzer.perturbation_explore_analyzer import perturbation_explore_analyzer_and_evaluation
from CANARY_SEFI.evaluator.analyzer.synthetical_analyzer import \
    model_security_synthetical_capability_analyzer_and_evaluation
from CANARY_SEFI.evaluator.analyzer.test_analyzer import model_capability_analyzer_and_evaluation, \
    attack_capability_analyzer_and_evaluation
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_all_adv_log
from CANARY_SEFI.evaluator.logger.attack_info_handler import find_attack_log, find_attack_log_by_name_and_base_model


def model_capability_test(dataset_info, model_list, model_config, img_proc_config):
    # 标记当前步骤
    global_system_log.set_step(Step.MODEL_CAPABILITY_TEST)

    def function(model_name, model_args, img_proc_args):
        # 模型基线测试
        msg = "在模型 {} 上运行模型能力评估".format(model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)
        inference(dataset_info, model_name, model_args, img_proc_args)

    BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, function)


def model_security_synthetical_capability_evaluation(batch_id, model_list):
    # 标记当前步骤
    global_system_log.set_step(Step.MODEL_SECURITY_SYNTHETICAL_CAPABILITY_EVALUATION)
    for model_name in model_list:
        model_security_synthetical_capability_analyzer_and_evaluation(batch_id, model_name)


def model_capability_evaluation(batch_id, model_list):
    # 标记当前步骤
    global_system_log.set_step(Step.MODEL_CAPABILITY_EVALUATION)
    for model_name in model_list:
        model_capability_analyzer_and_evaluation(batch_id, model_name)


def attack_capability_evaluation(batch_id, attacker_list):
    # 标记当前步骤
    global_system_log.set_step(Step.ATTACK_CAPABILITY_EVALUATION)
    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            attack_capability_analyzer_and_evaluation(batch_id, atk_name, base_model)



def adv_img_build_and_evaluation(dataset_info, attacker_list, attacker_config, model_config, img_proc_config):
    # 标记当前步骤
    global_system_log.set_step(Step.ADV_IMG_BUILD_AND_EVALUATION)

    def function(atk_name, atk_args, model_name, model_args, img_proc_args):
        msg = "攻击方法 {} 在模型 {} 上生成对抗样本并运行对抗样本质量评估".format(atk_name, model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)
        build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


def explore_attack_perturbation(dataset_info, attacker_list, attacker_config, model_config, img_proc_config,
                                explore_perturbation_config):
    # 标记当前步骤
    global_system_log.set_step(Step.EXPLORE_ATTACK_PERTURBATION)

    def function(atk_name, atk_args, model_name, model_args, img_proc_args):
        msg = "攻击方法 {} 在模型 {} 上 依据扰动上下限分步生成上述样本的对抗样本并运行扰动测评"\
            .format(atk_name, model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

        explore_perturbation_args = explore_perturbation_config.get(atk_name, None)
        explore_perturbation(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                             explore_perturbation_args)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


def attack_capability_test(batch_id, attacker_list, model_config, img_proc_config, transfer_attack_test="NOT",
                           transfer_attack_test_on_model_list = None):
    # 标记当前步骤
    global_system_log.set_step(Step.ATTACK_CAPABILITY_TEST)

    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:

            test_on_model_list = []
            if transfer_attack_test == "NOT":
                test_on_model_list.append(base_model)
            elif transfer_attack_test == "APPOINT":
                test_on_model_list = transfer_attack_test_on_model_list[atk_name][base_model]
            elif transfer_attack_test == "SELF_CROSS":
                test_on_model_list = attacker_list[atk_name]

            def function(model_name, model_args, img_proc_args):

                # 攻击测试
                msg = "攻击方法 {} (基于 {} 模型) 生成的对抗样本 针对 {} 模型进行攻击测试".format(atk_name, base_model, model_name)
                reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

                attack_log = find_attack_log_by_name_and_base_model(batch_id, atk_name, base_model)
                adv_inference(batch_id, attack_log, model_name, model_args, img_proc_args)

            BatchListIterator.model_list_iterator(test_on_model_list, model_config, img_proc_config, function)


def explore_perturbation_attack_capability_test(batch_id, attacker_list, model_config, img_proc_config):
    # 标记当前步骤
    global_system_log.set_step(Step.EXPLORE_ATTACK_PERTURBATION_ATTACK_TEST)

    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            def function(model_name, model_args, img_proc_args):
                # 攻击测试
                msg = "攻击方法 {} (基于 {} 模型) 在不同扰动限制下 生成的对抗样本 针对 {} 模型进行攻击测试".format(atk_name, base_model, model_name)
                reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

                attack_logs = find_attack_log_by_name_and_base_model(batch_id, atk_name, base_model, explore_perturbation_mode = True)

                for attack_log in attack_logs:
                    adv_inference(batch_id, attack_log, model_name, model_args, img_proc_args)

            BatchListIterator.model_list_iterator([base_model], model_config, img_proc_config, function)


def explore_perturbation_attack_capability_evaluation(batch_id, attacker_list):
    # 标记当前步骤
    global_system_log.set_step(Step.EXPLORE_ATTACK_PERTURBATION_ATTACK_EVALUATION)
    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            perturbation_explore_analyzer_and_evaluation(batch_id, atk_name, base_model)