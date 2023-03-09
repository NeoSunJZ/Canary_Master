import numpy as np
import torch
from matplotlib import pyplot as plt

from CANARY_SEFI.evaluator.monitor.grad_crm import ActivationsAndGradients, GradCAM
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.component.component_builder import build_dict_with_json_args, get_model
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import save_inference_test_data
from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class InferenceDetector:
    def __init__(self, inference_model_name, model_args, img_proc_args, run_device=None):

        self.model = get_model(inference_model_name, model_args, run_device)
        if self.model is None:
            # 未找到指定的Model
            raise RuntimeError("[ Config Error ] No model find, please check MODEL NAME")
        model_component = SEFI_component_manager.model_list.get(inference_model_name)

        # 注册Model的HOOK钩子以使用GRAD-CRM分析可解释性
        target_layers_getter = model_component.get("target_layers_getter")
        target_layers, reshape_transform = target_layers_getter(self.model)
        if target_layers_getter is not None:
            self.activations_and_grads = ActivationsAndGradients(target_layers=target_layers, reshape_transform=reshape_transform)

        # 预测器
        self.inference_detector = model_component.get("inference_detector")
        if self.inference_detector is None:
            # 未找到指定的预测器
            raise RuntimeError("[ Config Error ] Model find but inference detector is not existed ,please check your config")
        # 图片处理参数JSON转DICT
        self.img_proc_args_dict = build_dict_with_json_args(model_component, "img_processing", img_proc_args, run_device)
        # 图片预处理
        self.img_preprocessor = model_component.get("img_preprocessor")
        # 结果处理
        self.result_postprocessor = model_component.get("result_postprocessor")

        self.imgs = None

    def inference_detector_4_img(self, ori_imgs):
        inference_detector_func = self.inference_detector.get('func')

        # 图片预处理
        self.imgs = ori_imgs
        if self.img_preprocessor is not None:
            self.imgs = self.img_preprocessor(ori_imgs, self.img_proc_args_dict)

        if self.activations_and_grads is not None:
            self.activations_and_grads.gradients = []
            self.activations_and_grads.activations = []

        # 预测(关闭预测时torch的梯度，因为预测无需反向传播)
        self.model.eval()
        logits = inference_detector_func(self.model, self.imgs)
        self.model.zero_grad()

        if self.result_postprocessor is not None:
            result = self.result_postprocessor(logits, self.img_proc_args_dict)
        else:
            result = None

        check_cuda_memory_alloc_status(empty_cache=True)
        return result, logits


def inference_detector_4_img_batch(inference_model_name, model_args, img_proc_args, dataset_info,
                                   each_img_finish_callback=None, batch_size=1, completed_num=0, run_device=None):
    img_log_id_list = []
    inference_detector = InferenceDetector(inference_model_name, model_args, img_proc_args, run_device)

    def inference_iterator(imgs, img_log_ids, img_labels):
        # 执行预测
        result, logits = inference_detector.inference_detector_4_img(imgs)
        inference_labels, inference_conf_arrays = result[0], result[1]

        if inference_detector.activations_and_grads is not None:
            cam = GradCAM(activations_and_grads=inference_detector.activations_and_grads)
            grayscale_cams_with_true_labels = cam(output=logits, input_tensor=inference_detector.imgs, target_category=img_labels)
            grayscale_cams_with_inference_labels = cam(output=logits, input_tensor=inference_detector.imgs, target_category=inference_labels)
        else:
            grayscale_cams_with_true_labels = None
            grayscale_cams_with_inference_labels = None

        # batch分割
        for index in range(len(inference_labels)):
            img_log_id_list.append(img_log_ids[index])
            if each_img_finish_callback is not None:
                each_img_finish_callback(imgs[index], inference_labels[index])

            # 写入必要日志
            save_inference_test_data(img_log_ids[index], dataset_info.dataset_type.value, inference_model_name,
                                     inference_labels[index], inference_conf_arrays[index],
                                     (grayscale_cams_with_true_labels[index], grayscale_cams_with_inference_labels[index]))

    dataset_image_reader(inference_iterator, dataset_info, batch_size, completed_num)
    task_manager.sys_log_logger.update_finish_status(True)

    check_cuda_memory_alloc_status(empty_cache=True)
    return img_log_id_list
