import torch
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args, get_model
from CANARY_SEFI.core.function.dataset_function import dataset_image_reader
from CANARY_SEFI.evaluator.logger.inference_logger import InferenceLogger


class InferenceDetector:
    def __init__(self, model_name, model_args, img_proc_args):
        # 测评日志
        self.logger = InferenceLogger(model_name)

        self.model = get_model(model_name, model_args)
        if self.model is None:
            # 未找到指定的Model
            raise RuntimeError("No model find, please check MODEL NAME")
        model_component = SEFI_component_manager.model_list.get(model_name)
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

        # 预测
        result = inference_detector_func(self.model, img)

        if self.result_postprocessor is not None:
            result = self.result_postprocessor(result, self.img_proc_args_dict)

        torch.cuda.empty_cache()
        return result


def inference_detector_4_img_batch(batch_token, model_name, model_args, img_proc_args, dataset_name, dataset_size=None,
                           img_list=None, each_img_finish_callback=None):
    label_list = []
    inference_detector = InferenceDetector(model_name, model_args, img_proc_args)

    def inference_iterator(img, img_name, img_label):
        # 执行攻击
        label, conf_array = inference_detector.inference_detector_4_img(img)
        # 保存至临时文件夹
        label_list.append(label)
        if each_img_finish_callback is not None:
            each_img_finish_callback(img, label)

        # 写入必要日志
        inference_detector.logger.img_name = img_name
        inference_detector.logger.ori_label = img_label
        inference_detector.logger.inference_label = label
        inference_detector.logger.inference_conf_array = conf_array
        inference_detector.logger.next(batch_token)

    dataset_image_reader(batch_token, inference_iterator, dataset_name, dataset_size, img_list)
    torch.cuda.empty_cache()
    return label_list
