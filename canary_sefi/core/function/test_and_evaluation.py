from colorama import Fore

from canary_sefi.core.function.adversarial_training import adv_training
from canary_sefi.core.function.basic.img_trans_function import adv_trans_4_img_batch
from canary_sefi.core.function.comparative_test_adv_example import adv_comparative_test
from canary_sefi.entity.dataset_info_entity import DatasetType
from canary_sefi.evaluator.analyzer.inference_data_analyzer import defense_normal_effectiveness_analyzer_and_evaluation, \
    analyzer_log_handler, defense_adv_effectiveness_analyzer_and_evaluation
from canary_sefi.core.function.enum.test_level_enum import TestLevel
from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.enum.step_enum import Step
from canary_sefi.core.function.enum.transfer_attack_type_enum import TransferAttackType
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.core.function.inference import inference, adv_inference
from canary_sefi.core.function.generate_adv_example import build_AEs, build_AEs_with_perturbation_increment
from canary_sefi.core.function.helper.batch_list_iterator import BatchListIterator
from canary_sefi.evaluator.analyzer.synthetical_analyzer import \
    model_security_synthetical_capability_analyzer_and_evaluation, attack_synthetical_capability_analyzer_and_evaluation
from canary_sefi.evaluator.analyzer.test_data_analyzer import model_inference_capability_analyzer_and_evaluation, \
    attack_deflection_capability_analyzer_and_evaluation, \
    attack_capability_with_perturbation_increment_analyzer_and_evaluation, \
    attack_adv_example_da_and_cost_analyzer_and_evaluation
from canary_sefi.evaluator.logger.attack_info_handler import find_attack_log_by_name_and_base_model, \
    find_attack_log_by_name


# 生成对抗样本
def adv_example_generate(dataset_info, attacker_list, attacker_config, model_config, img_proc_config,
                         adv_example_generate_batch_config):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ADV_EXAMPLE_GENERATE)

    def function(atk_name, atk_args, model_name, model_args, img_proc_args, run_device):
        msg = "[Device {}] Generating Adv Example By Attack Method {} on(base) Model {}.".format(run_device, atk_name,
                                                                                                 model_name)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
        build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                  atk_batch_config=adv_example_generate_batch_config, run_device=run_device)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


# 攻击方法生成对抗样本综合对比测试(图像相似性/模型注意力差异对比/像素差异对比)
def attack_adv_example_comparative_test(attacker_list, dataset_info, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_ADV_EXAMPLE_COMPARATIVE_TEST)
    for atk_name in attacker_list:
        attack_logs = find_attack_log_by_name(atk_name)
        for attack_log in attack_logs:
            msg = "Test Adv Example(Generated by Attack Method({} *Base Model {}*)(Use Numpy Array Data:{}))'s " \
                  "Disturbance-aware / Gradient-weighted Class Activation Mapping Comparative / Pixel Difference" \
                .format(atk_name, attack_log["base_model"], use_raw_nparray_data)
            reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

            adv_comparative_test(attack_log, dataset_info, use_raw_nparray_data)


# 攻击方法生成对抗样本图像相似性(扰动距离)/生成代价评估
def attack_adv_example_da_and_cost_evaluation(attacker_list, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_ADV_EXAMPLE_DA_AND_COST_EVALUATION)
    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            attack_adv_example_da_and_cost_analyzer_and_evaluation(atk_name, base_model, use_raw_nparray_data)


# 攻击偏转能力测试
def attack_deflection_capability_test(attacker_list, dataset_info, model_config, img_proc_config,
                                      inference_batch_config,
                                      transfer_attack_test=TransferAttackType.NOT,
                                      transfer_attack_test_on_model_list=None, use_raw_nparray_data=False,
                                      transfer_test_level=TestLevel.ESSENTIAL_ONLY):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_DEFLECTION_CAPABILITY_TEST)

    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:

            test_on_model_list = []
            if transfer_attack_test == TransferAttackType.NOT:
                test_on_model_list.append(base_model)
            elif transfer_attack_test == TransferAttackType.APPOINT:
                test_on_model_list = transfer_attack_test_on_model_list[atk_name][base_model].copy()
            elif transfer_attack_test == TransferAttackType.APPOINT_SIMPLE:
                test_on_model_list = transfer_attack_test_on_model_list[atk_name].copy()
                test_on_model_list.append(base_model)
            elif transfer_attack_test == TransferAttackType.SELF_CROSS:
                test_on_model_list = attacker_list[atk_name]

            def function(model_name, model_args, img_proc_args, run_device):
                # 攻击测试
                msg = "Inferencing Adv Example(Generated by Attack Method({} *Base Model {}*)" \
                      "(Use Numpy Array Data:{}))'s Label by Model {} " \
                    .format(atk_name, base_model, use_raw_nparray_data, model_name)
                reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

                test_level = TestLevel.FULL
                if model_name != base_model:
                    msg += "(Transferability Test, Level:{})".format(transfer_test_level)
                    test_level = transfer_test_level

                attack_log = find_attack_log_by_name_and_base_model(atk_name, base_model)
                adv_inference(dataset_info, attack_log, model_name, model_args, img_proc_args,
                              inference_batch_config=inference_batch_config,
                              use_raw_nparray_data=use_raw_nparray_data,
                              run_device=run_device,
                              test_level=test_level)

            BatchListIterator.model_list_iterator(test_on_model_list, model_config, img_proc_config, function)


# 攻击方法推理偏转效果/模型注意力偏转效果评估
def attack_deflection_capability_evaluation(attacker_list, dataset_info=None, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_DEFLECTION_CAPABILITY_EVALUATION)
    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            attack_deflection_capability_analyzer_and_evaluation(atk_name, base_model, dataset_info,
                                                                 use_raw_nparray_data)


# 模型推理能力测试
def model_inference_capability_test(dataset_info, model_list, model_config, img_proc_config, inference_batch_config,
                                    ):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.MODEL_INFERENCE_CAPABILITY_TEST)

    def function(model_name, model_args, img_proc_args, run_device):
        # 模型基线测试
        msg = "Inferencing Img Label by Model {}".format(model_name)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
        inference(dataset_info, model_name, model_args, img_proc_args, inference_batch_config,
                  run_device=run_device)

    BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, function)


# 模型推理能力评估
def model_inference_capability_evaluation(model_list):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.MODEL_INFERENCE_CAPABILITY_EVALUATION)
    for model_name in model_list:
        model_inference_capability_analyzer_and_evaluation(model_name)


def defense_model_normal_inference_capability_evaluation(dataset_info, model_list, defense_model_list):
    task_manager.sys_log_logger.set_step(Step.DEFENSE_NORMAL_EFFECTIVENESS_EVALUATION)
    for model_name in model_list:
        base_normal_analyzer_log = analyzer_log_handler(model_name, DatasetType.NORMAL.value)
        for defense_name in defense_model_list.get(model_name):
            defense_normal_effectiveness_analyzer_and_evaluation(base_normal_analyzer_log, dataset_info, defense_name,
                                                                 model_name)


def defense_model_adv_inference_capability_evaluation(attacker_list, attacker_config, model_config, img_proc_config,
                                                      defense_model_list, use_raw_nparray_data=False):
    task_manager.sys_log_logger.set_step(Step.DEFENSE_ADVERSARIAL_EFFECTIVENESS_EVALUATION)

    def function(atk_name, atk_args, model_name, model_args, img_proc_args, run_device):
        for defense_name in defense_model_list.get(model_name):
            msg = "Evaluate Defense Method {} on Model {} with Adversarial Examples Generated by {}".format(
                defense_name, model_name, atk_name)
            reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
            defense_adv_effectiveness_analyzer_and_evaluation(atk_name, defense_name, model_name, use_raw_nparray_data)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


# 模型综合能力评估
def model_security_synthetical_capability_evaluation(model_list, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.MODEL_SECURITY_SYNTHETICAL_CAPABILITY_EVALUATION)
    for model_name in model_list:
        model_security_synthetical_capability_analyzer_and_evaluation(model_name, use_raw_nparray_data)


# 攻击算法综合能力评估
def attack_synthetical_capability_evaluation(attacker_list, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_SYNTHETICAL_CAPABILITY_EVALUATION)
    for attack_name in attacker_list:
        attack_synthetical_capability_analyzer_and_evaluation(attack_name, use_raw_nparray_data)


# 生成对抗样本(扰动递增)
def adv_example_generate_with_perturbation_increment(dataset_info, attacker_list, attacker_config, model_config,
                                                     img_proc_config, adv_example_generate_batch_config,
                                                     perturbation_increment_config):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ADV_EXAMPLE_GENERATE_WITH_PERTURBATION_INCREMENT)

    def function(atk_name, atk_args, model_name, model_args, img_proc_args, run_device):
        msg = "Generating Adv Example By Attack Method {} on(base) Model {} in the case of increasing perturbations.".format(
            atk_name, model_name)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

        perturbation_increment_args = perturbation_increment_config.get(atk_name, None)
        build_AEs_with_perturbation_increment(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                                              atk_batch_config=adv_example_generate_batch_config,
                                              perturbation_increment_args=perturbation_increment_args,
                                              run_device=run_device)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


# 攻击偏转能力测试(扰动递增)
def attack_deflection_capability_test_with_perturbation_increment(dataset_info, attacker_list, model_config, img_proc_config,
                                                                  use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_DEFLECTION_CAPABILITY_TEST_WITH_PERTURBATION_INCREMENT)

    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            def function(model_name, model_args, img_proc_args, run_device):
                # 攻击测试
                msg = "Inferencing Adv Example(Generated by Attack Method({} *Base Model {}*)" \
                      "(Use Numpy Array Data:{}) in the case of increasing perturbations)'s Label by Model {}" \
                    .format(atk_name, base_model, use_raw_nparray_data, model_name)
                reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

                attack_logs = find_attack_log_by_name_and_base_model(atk_name, base_model,
                                                                     perturbation_increment_mode=True)

                for attack_log in attack_logs:
                    adv_inference(dataset_info, attack_log, model_name, model_args, img_proc_args, use_raw_nparray_data,
                                  run_device=run_device)

            BatchListIterator.model_list_iterator([base_model], model_config, img_proc_config, function)


# 攻击对抗样本对比测试（质量劣化（DA）测试，模型注意力图像对比，像素差异对比）(扰动递增)
def attack_adv_example_da_test_with_perturbation_increment(attacker_list, dataset_info, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_ADV_EXAMPLE_COMPARATIVE_TEST_WITH_PERTURBATION_INCREMENT)

    for atk_name in attacker_list:
        attack_logs = find_attack_log_by_name(atk_name)
        for attack_log in attack_logs:
            msg = "Test Adv Example(Generated by Attack Method({} *Base Model {}*)" \
                  "(Use Numpy Array Data:{}) in the case of increasing perturbations)'s Disturbance-aware" \
                .format(atk_name, attack_log["base_model"], use_raw_nparray_data)
            reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

            adv_comparative_test(attack_log, dataset_info, use_raw_nparray_data)


# 攻击对抗样本质量\攻击偏转能力(扰动递增)评估
def attack_capability_evaluation_with_perturbation_increment(attacker_list, dataset_info=None,
                                                             use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ATTACK_EVALUATION_WITH_PERTURBATION_INCREMENT)
    for atk_name in attacker_list:
        for base_model in attacker_list[atk_name]:
            attack_capability_with_perturbation_increment_analyzer_and_evaluation(atk_name, base_model, dataset_info,
                                                                                  use_raw_nparray_data)


def adv_trans_generate(trans_list, trans_config, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.ADV_TRANS_GENERATE)
    for atk_name in trans_list:
        for trans_name in trans_list[atk_name]:
            for base_model in trans_list[atk_name][trans_name]:
                trans_args = trans_config.get(trans_name, None)
                attack_log = find_attack_log_by_name_and_base_model(atk_name, base_model)
                adv_trans_4_img_batch(trans_name=trans_name, trans_args=trans_args, atk_log=attack_log,
                                      use_raw_nparray_data=use_raw_nparray_data)


# 防御样本攻击偏转能力测试
def trans_deflection_capability_test(dataset_info, trans_list, model_config, img_proc_config,
                                     inference_batch_config,
                                     transfer_attack_test=TransferAttackType.NOT,
                                     transfer_attack_test_on_model_list=None, use_raw_nparray_data=False,
                                     transfer_test_level=TestLevel.ESSENTIAL_ONLY):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.TRANS_DEFLECTION_CAPABILITY_TEST)

    for atk_name in trans_list:
        for trans_name in trans_list[atk_name]:
            for base_model in trans_list[atk_name][trans_name]:

                test_on_model_list = []
                test_on_model_list.append(base_model)

                # if transfer_attack_test == TransferAttackType.NOT:
                #     test_on_model_list.append(base_model)
                # elif transfer_attack_test == TransferAttackType.APPOINT:
                #     test_on_model_list = transfer_attack_test_on_model_list[atk_name][base_model].copy()
                # elif transfer_attack_test == TransferAttackType.SELF_CROSS:
                #     test_on_model_list = attacker_list[atk_name].copy()

                def function(model_name, model_args, img_proc_args, run_device):
                    # 攻击测试
                    msg = "Inferencing Trans(Generated by Attack Method({} *Base Model {}* Trans Method {})" \
                          "(Use Numpy Array Data:{}))'s Label by Model {} " \
                        .format(atk_name, base_model, trans_name, use_raw_nparray_data, model_name)
                    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

                    test_level = TestLevel.FULL
                    if model_name != base_model:
                        msg += "(Transferability Test, Level:{})".format(transfer_test_level)
                        test_level = transfer_test_level

                    attack_log = find_attack_log_by_name_and_base_model(atk_name, base_model)

                    adv_inference(dataset_info, attack_log, model_name, model_args, img_proc_args,
                                  inference_batch_config=inference_batch_config,
                                  use_raw_nparray_data=use_raw_nparray_data, run_device=run_device,
                                  test_level=test_level, trans_name=trans_name)

                BatchListIterator.model_list_iterator(test_on_model_list, model_config, img_proc_config, function)


# 防御样本-攻击方法推理偏转效果/模型注意力偏转效果评估
def trans_deflection_capability_evaluation(trans_list, dataset_info=None, use_raw_nparray_data=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.TRANS_DEFLECTION_CAPABILITY_EVALUATION)
    for atk_name in trans_list:
        for trans_name in trans_list[atk_name]:
            for base_model in trans_list[atk_name][trans_name]:
                attack_deflection_capability_analyzer_and_evaluation(atk_name, base_model, dataset_info,
                                                                     use_raw_nparray_data, trans_name)


def adv_defense_training(dataset_info, defense_list, defense_config, model_config, img_proc_config):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.DEFENSE_ADVERSARIAL_TRAINING)

    def function(defense_name, defense_args, model_name, model_args, img_proc_args, run_device):
        msg = "[Device {}] Adversarial Training With Defense Method ({}) on Model ({}).".format(run_device,
                                                                                                defense_name,
                                                                                                model_name)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
        adv_training(dataset_info, defense_name, defense_args, model_name, model_args, img_proc_args,
                     run_device=run_device)

    BatchListIterator.defense_list_iterator(defense_list, defense_config, model_config, img_proc_config, function)
