import numpy as np
from colorama import Fore
from tqdm.auto import tqdm

from CANARY_SEFI.core.component.component_builder import get_model
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.function.basic.attacker_function import AdvAttacker
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader
from CANARY_SEFI.core.function.init_dataset import init_dataset
from CANARY_SEFI.handler.image_handler.img_io_handler import save_pic_to_temp
from CANARY_SEFI.task_manager import task_manager


class CorrectnessCheck:
    def __init__(self, config):
        print(Fore.RED + "\n")
        for i in range(4):
            print(" WARNING : START WITH CORRECTNESS CHECK MODE ")
        task_manager.init_task(task_token="C_TEST", show_logo=True, run_device="cuda", not_retain_same_token=True, logo_color=Fore.RED)

        self.dataset_info = init_dataset(config.get("dataset"), config.get("dataset_size"), config.get("dataset_seed", None))

        self.model_name = config.get("model_name", None)
        self.atk_name = config.get("atk_name", None)

        self.model_args = config.get("model_args", None)
        self.atk_args = config.get("atk_args", None)
        self.img_proc_args = config.get("img_proc_args", None)

        self.run_device = config.get("run_device", None)
        self.adv_example_generate_batch_size = config.get("adv_example_generate_batch_size", 1)

    def attack_method_correctness_test(self, start_num=0, save_adv_example=False):
        # 构建攻击者
        adv_attacker = AdvAttacker(self.atk_name, self.atk_args, self.model_name, self.model_args, self.img_proc_args, self.dataset_info, self.adv_example_generate_batch_size, self.run_device)

        model_component = SEFI_component_manager.model_list.get(self.model_name)
        inference_detector = model_component.get("inference_detector")
        result_postprocessor = model_component.get("result_postprocessor")
        img_reverse_processor = model_component.get("img_reverse_processor")  # 图片后处理
        img_preprocessor = model_component.get("img_preprocessor")  # 图片前处理

        model = get_model(self.model_name, self.model_args, self.run_device, model_query_logger=None)
        inference_detector_func = inference_detector.get('func')

        logits_ORI = []
        logits_ADV_O = []
        logits_ADV_NPY = []
        logits_ADV_IMG = []

        def img_post_handler(adv_imgs, args):
            # 输出的结果直接进行预测
            logits_ADV_O.append(inference_detector_func(model, adv_imgs))
            return img_reverse_processor(adv_imgs, args)

        if adv_attacker.img_reverse_processor is not None:
            adv_attacker.img_reverse_processor = img_post_handler

        # 攻击单图片迭代函数
        def attack_iterator(imgs, img_log_ids, img_labels, save_raw_data=True):
            # 原始图片预测
            img_preprocessed = img_preprocessor(imgs, self.model_args)
            logits_ORI.append(inference_detector_func(model, img_preprocessed))

            # 执行攻击
            adv_results, tlabels = adv_attacker.adv_attack_4_img(imgs, img_labels)

            # 输出的结果存为NPY后再预测
            img_preprocessed = img_preprocessor(adv_results, self.model_args)
            logits_ADV_NPY.append(inference_detector_func(model, img_preprocessed))

            # 输出的结果存为IMG后再预测
            img_preprocessed = img_preprocessor(np.clip(adv_results, 0, 255).astype(np.uint8), self.model_args)
            logits_ADV_IMG.append(inference_detector_func(model, img_preprocessed))


            # batch分割
            for index in range(len(adv_results)):

                adv_result = adv_results[index]
                img_log_id = img_log_ids[index]
                tlabel = tlabels[index] if tlabels is not None else None

                img_file_name = "adv_{}.png".format(img_log_id)
                raw_file_name = "adv_raw_{}.npy".format(img_log_id)
                if save_adv_example:
                    save_pic_to_temp("TEST" + "/", img_file_name, adv_result)
                    save_pic_to_temp("TEST" + "/", raw_file_name, adv_result, save_as_numpy_array=True)

                # 日志
                print("[Simulation Log] IMG_LOG_ID:{},IMG_FILE_NAME:{},RAW_FILE_NAME:{},TLABEL:{}".format(img_log_id, img_file_name, raw_file_name, tlabel))
                print("[Simulation Log] COST_TIME:{},QUERY_NUM:{}".format(float(adv_attacker.cost_time) / len(adv_results), adv_attacker.query_num))

        dataset_image_reader(attack_iterator, self.dataset_info, self.adv_example_generate_batch_size, start_num)
        adv_attacker.destroy()
        del adv_attacker

        # 处理结论
        inference_detector_func = inference_detector.get('func')

        result_ORI = []
        result_ADV_O = []
        result_ADV_NPY = []
        result_ADV_IMG = []

        for logits in logits_ORI:
            results, _ = result_postprocessor(logits, self.img_proc_args) if result_postprocessor is not None else None
            for result in results:
                result_ORI.append(result)

        for logits in logits_ADV_O:
            results, _ = result_postprocessor(logits, self.img_proc_args) if result_postprocessor is not None else None
            for result in results:
                result_ADV_O.append(result)

        for logits in logits_ADV_NPY:
            results, _ = result_postprocessor(logits, self.img_proc_args) if result_postprocessor is not None else None
            for result in results:
                result_ADV_NPY.append(result)

        for logits in logits_ADV_IMG:
            results, _ = result_postprocessor(logits, self.img_proc_args) if result_postprocessor is not None else None
            for result in results:
                result_ADV_IMG.append(result)

        print("result_ORI:{}".format(result_ORI))
        print("logits_ADV_O:{}".format(result_ADV_O))
        print("logits_ADV_NPY:{}".format(result_ADV_NPY))
        print("logits_ADV_IMG:{}".format(result_ADV_IMG))

        statistics = {
            "ADV_O_Mc": 0, "ADV_NPY_Mc": 0, "ADV_IMG_Mc": 0,
        }
        for index in range(len(result_ORI)):
            if result_ORI[index] != result_ADV_O[index]:
                statistics["ADV_O_Mc"] += 1
            if result_ORI[index] != result_ADV_NPY[index]:
                statistics["ADV_NPY_Mc"] += 1
            if result_ORI[index] != result_ADV_IMG[index]:
                statistics["ADV_IMG_Mc"] += 1
        print("MR: ADV_O_Mc:{} ADV_NPY_Mc:{} ADV_IMG_Mc:{}".format(statistics["ADV_O_Mc"]/len(result_ORI),
                                                                   statistics["ADV_NPY_Mc"]/len(result_ORI),
                                                                   statistics["ADV_IMG_Mc"]/len(result_ORI)))
