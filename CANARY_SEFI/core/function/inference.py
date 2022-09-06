from tqdm import tqdm

from CANARY_SEFI.core.function.basic.model_function import inference_detector_4_img_batch


def inference(dataset_info, model_name, model_args, img_proc_args):
    with tqdm(total=dataset_info.dataset_size, desc="推理进度", ncols=80) as bar:
        def each_img_finish_callback(img, result):
            bar.update(1)

        inference_detector_4_img_batch(model_name, model_args, img_proc_args, dataset_info, each_img_finish_callback)
