from matplotlib import pyplot as plt
from tqdm import tqdm

from CANARY_SEFI.evaluator.logger.inference_test_data_handler import get_inference_test_data_by_img_id
from CANARY_SEFI.handler.image_handler.img_utils import img_size_uniform_fix
from CANARY_SEFI.handler.image_handler.plt_handler import cam_diff_plt_builder, show_plt, img_diff_plt_builder
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader, dataset_single_image_reader
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo, DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_da_test_data_handler import save_adv_example_da_test_data
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_logs_by_attack_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.evaluator.tester.adv_disturbance_aware import AdvDisturbanceAwareTester
from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


def adv_comparative_test(atk_log, dataset_info, use_raw_nparray_data=False):
    all_adv_log = find_adv_example_file_logs_by_attack_id(atk_log['attack_id'])

    adv_img_cursor_list = []
    adv_img_ori_dict = {}
    for adv_log in all_adv_log:
        adv_img_cursor_list.append(adv_log["adv_img_file_id"])
        adv_img_ori_dict[adv_log["adv_img_file_id"]] = adv_log["ori_img_id"]

    adv_dataset_info = DatasetInfo(None, None, None, adv_img_cursor_list)
    adv_dataset_info.dataset_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA if use_raw_nparray_data else DatasetType.ADVERSARIAL_EXAMPLE_IMG

    with tqdm(total=adv_dataset_info.dataset_size, desc="Adv-example Disturbance-aware Test Progress", ncols=120) as bar:
        participant = "{}({})".format(atk_log['atk_name'], atk_log['base_model'])
        if atk_log['atk_perturbation_budget'] != "None":
            participant = "{}({})(e-{})".format(atk_log['atk_name'], atk_log['base_model'],
                                                   str(round(float(atk_log['atk_perturbation_budget']), 5)))
        participant += "(RAW)" if use_raw_nparray_data else "(IMG)"
        is_skip, completed_num = global_recovery.check_skip(participant)
        if is_skip:
            return None

        adv_da_tester = AdvDisturbanceAwareTester()

        ori_img_list = {}

        def adv_img_iterator(adv_img, adv_img_file_id, img_label):
            ori_img_id = adv_img_ori_dict[adv_img_file_id[0]]
            # 尝试在缓存中找到原始图片
            ori_img_data = ori_img_list.get(ori_img_id, None)
            if ori_img_data is None:
                ori_img_log = find_img_log_by_id(ori_img_id)
                ori_img, ori_label = dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])
                # 存入临时缓存
                ori_img_list[ori_img_id] = {
                    "ori_img": ori_img,
                    "ori_label": ori_label,
                }
            else:
                ori_img = ori_img_data.get("ori_img")
                ori_label = ori_img_data.get("ori_label")

            ori, adv = img_size_uniform_fix(ori_img, adv_img[0])
            # 执行Disturbance-Aware测试
            adv_da_test_result = adv_da_tester.test_all(ori, adv)
            # 执行像素差异对比分析
            diff_plt = img_diff_plt_builder(ori, adv)
            show_plt(diff_plt)

            # 写入日志
            save_adv_example_da_test_data(adv_img_file_id[0], adv_dataset_info.dataset_type.value, adv_da_test_result)
            task_manager.sys_log_logger.update_completed_num(1)
            bar.update(1)

        dataset_image_reader(adv_img_iterator, adv_dataset_info, batch_size=1, completed_num=completed_num)
        task_manager.sys_log_logger.update_finish_status(True)
        check_cuda_memory_alloc_status(empty_cache=True)




