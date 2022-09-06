from colorama import Fore
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.core.function.inference import inference
from CANARY_SEFI.core.function.make_adv_example import build_AEs, explore_perturbation
from CANARY_SEFI.core.function.helper.batch_list_iterator import BatchListIterator
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo, DatasetType
from CANARY_SEFI.evaluator.logger.adv_logger import find_batch_adv_log
from CANARY_SEFI.evaluator.logger.attack_logger import find_attack_log


def model_capability_test(dataset_info, model_list, model_config, img_proc_config):
    # 标记当前步骤
    global_system_log.set_step("MODEL_CAPABILITY_TEST")

    def function(model_name, model_args, img_proc_args):
        # 模型基线测试
        msg = "在模型 {} 上运行模型能力评估".format(model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)
        inference(dataset_info, model_name, model_args, img_proc_args)

    BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, function)


def adv_img_build_and_evaluation(dataset_info, attacker_list, attacker_config, model_config,
                                 img_proc_config):
    # 标记当前步骤
    global_system_log.set_step("ADV_IMG_BUILD_AND_EVALUATION")

    def function(atk_name, atk_args, model_name, model_args, img_proc_args):
        msg = "基于攻击方法 {} 在模型 {} 上生成上述样本的对抗样本并运行对抗样本质量评估" \
            .format(atk_name, model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)
        build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


def explore_attack_perturbation(dataset_info, attacker_list, attacker_config, model_config, img_proc_config,
                                explore_perturbation_config):
    # 标记当前步骤
    global_system_log.set_step("EXPLORE_ATTACK_PERTURBATION")

    def function(atk_name, atk_args, model_name, model_args, img_proc_args):
        msg = "基于攻击方法 {} 在模型 {} 上 依据扰动上下限分步生成上述样本的对抗样本并运行扰动测评"\
            .format(atk_name, model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

        explore_perturbation_args = explore_perturbation_config.get(atk_name, None)
        explore_perturbation(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                             explore_perturbation_args)

    BatchListIterator.attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function)


def attack_capability_test(batch_id, model_list, model_config, img_proc_config, full_adv_transfer_test=False):
    # 标记当前步骤
    global_system_log.set_step("ATTACK_CAPABILITY_TEST")

    # 验证攻击效果
    # 获取当前批次全部攻击图片目录
    all_adv_log = find_batch_adv_log(batch_id)

    def function(model_name, model_args, img_proc_args):
        msg = "基于对抗样本在模型 {} 上运行模型能力评估".format(model_name)
        reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

        adv_img_cursor_list = []
        for adv_log in all_adv_log:
            if full_adv_transfer_test:
                adv_img_cursor_list.append(adv_log[0])
            else:
                attack_log = find_attack_log(adv_log[2])[0]
                if attack_log[3] == model_name:
                    adv_img_cursor_list.append(adv_log[0])

        adv_dataset_info = DatasetInfo(None, None, None, adv_img_cursor_list)
        adv_dataset_info.dataset_type = DatasetType.ADVERSARIAL_EXAMPLE

        inference(adv_dataset_info, model_name, model_args, img_proc_args)

    BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, function)
