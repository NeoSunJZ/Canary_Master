import torch

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args, get_model
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import save_inference_test_data
from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class InferenceDetector:
    def __init__(self, inference_model_name, model_args, img_proc_args):
        # 测评日志

        self.model = get_model(inference_model_name, model_args)
        if self.model is None:
            # 未找到指定的Model
            raise RuntimeError("No model find, please check MODEL NAME")
        model_component = SEFI_component_manager.model_list.get(inference_model_name)
        # 预测器
        self.inference_detector = model_component.get("inference_detector")
        if self.inference_detector is None:
            # 未找到指定的预测器
            raise RuntimeError("Model find but inference detector is not existed ,please check your config")
        # 图片处理参数JSON转DICT
        self.img_proc_args_dict = build_dict_with_json_args(model_component, "img_processing", img_proc_args)
        # 图片预处理
        self.img_preprocessor = model_component.get("img_preprocessor")
        # 结果处理
        self.result_postprocessor = model_component.get("result_postprocessor")

    def inference_detector_4_img(self, ori_img):
        inference_detector_func = self.inference_detector.get('func')

        # 图片预处理
        img = ori_img
        if self.img_preprocessor is not None:
            img = self.img_preprocessor(ori_img, self.img_proc_args_dict)

        # 预测(关闭预测时torch的梯度，因为预测无需反向传播)
        with torch.no_grad():
            result = inference_detector_func(self.model, img)

        if self.result_postprocessor is not None:
            result = self.result_postprocessor(result, self.img_proc_args_dict)

        check_cuda_memory_alloc_status(empty_cache=True)
        return result


def inference_detector_4_img_batch(inference_model_name, model_args, img_proc_args, dataset_info, each_img_finish_callback=None, completed_num=0):

    img_log_id_list = []
    inference_detector = InferenceDetector(inference_model_name, model_args, img_proc_args)

    def inference_iterator(img, img_id, img_label):
        # 执行预测
        label, conf_array = inference_detector.inference_detector_4_img(img)
        img_log_id_list.append(img_id)
        if each_img_finish_callback is not None:
            each_img_finish_callback(img, label)

        # 写入必要日志
        save_inference_test_data(img_id, dataset_info.dataset_type.value, inference_model_name, label, conf_array)

    dataset_image_reader(inference_iterator, dataset_info, completed_num)
    batch_manager.sys_log_logger.update_finish_status(True)

    check_cuda_memory_alloc_status(empty_cache=True)
    return img_log_id_list
