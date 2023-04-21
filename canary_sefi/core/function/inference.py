from tqdm import tqdm

from canary_sefi.core.function.basic.model_function import inference_detector_4_img_batch
from canary_sefi.core.function.enum.test_level_enum import TestLevel
from canary_sefi.core.function.helper.recovery import global_recovery
from canary_sefi.entity.dataset_info_entity import DatasetInfo, DatasetType
from canary_sefi.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id
from canary_sefi.evaluator.logger.trans_file_info_handler import find_adv_trans_file_logs_by_attack_id_and_trans_name


def inference(dataset_info, model_name, model_args, img_proc_args, inference_batch_config, run_device=None):
    with tqdm(total=dataset_info.dataset_size, desc="Inference Progress", ncols=120) as bar:
        def each_img_finish_callback(img, result):
            bar.update(1)

        is_skip, completed_num = global_recovery.check_skip(model_name)
        if is_skip:
            return None
        batch_size = inference_batch_config.get(model_name, 1)
        inference_detector_4_img_batch(model_name, model_args, img_proc_args, dataset_info,
                                       each_img_finish_callback=each_img_finish_callback,
                                       batch_size=batch_size,
                                       completed_num=completed_num,
                                       run_device=run_device)


def adv_inference(dataset_info, atk_log, test_model, model_args, img_proc_args, inference_batch_config,
                  use_raw_nparray_data=False, run_device=None,
                  test_level=TestLevel.FULL, trans_name=None):
    if trans_name is not None:
        all_adv_log = find_adv_trans_file_logs_by_attack_id_and_trans_name(atk_log['attack_id'], trans_name)
        dataset_type = DatasetType.TRANSFORM_RAW_DATA if use_raw_nparray_data else DatasetType.TRANSFORM_IMG
    else:
        all_adv_log = find_adv_example_file_logs_by_attack_id(atk_log['attack_id'])
        dataset_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG

    adv_img_cursor_list = []
    for adv_log in all_adv_log:
        adv_img_cursor_list.append(adv_log["adv_img_file_id"] if trans_name is None else adv_log["adv_trans_img_file_id"])

    adv_dataset_info = DatasetInfo(
        dataset_name=None,
        dataset_extra_info={
            "is_gray": dataset_info.is_gray,
        },
        dataset_type=dataset_type,
        dataset_seed=None,
        dataset_size=None,
        img_cursor_list=adv_img_cursor_list)
    if trans_name is None:
        adv_dataset_info.dataset_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG
    else:
        adv_dataset_info.dataset_type = DatasetType.TRANSFORM_RAW_DATA if use_raw_nparray_data else DatasetType.TRANSFORM_IMG

    desc = "Adv-example Inference Progress" if trans_name is None else "Trans-example Inference Progress"
    with tqdm(total=adv_dataset_info.dataset_size, desc=desc, ncols=120) as bar:
        def each_img_finish_callback(img, result):
            bar.update(1)

        participant = "{}({}):{}".format(atk_log['atk_name'], atk_log['base_model'], test_model)
        if atk_log['atk_perturbation_budget'] != "None" and atk_log['atk_perturbation_budget'] is not None:
            participant = "{}({})(e-{}):{}".format(atk_log['atk_name'], atk_log['base_model'],
                                                   str(round(float(atk_log['atk_perturbation_budget']), 5)), test_model)
        if trans_name is not None:
            participant += ":{}".format(trans_name)
        participant += "(RAW)" if use_raw_nparray_data else "(IMG)"
        is_skip, completed_num = global_recovery.check_skip(participant)
        if is_skip:
            return None

        batch_size = inference_batch_config.get(test_model, 1)
        inference_detector_4_img_batch(test_model, model_args, img_proc_args, adv_dataset_info,
                                       each_img_finish_callback=each_img_finish_callback,
                                       batch_size=batch_size,
                                       completed_num=completed_num,
                                       run_device=run_device,
                                       test_level=test_level)
