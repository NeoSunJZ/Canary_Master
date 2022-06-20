from tqdm import tqdm

from CANARY_SEFI.core.function.model_function import inference_detector_4_img_batch


def clear_inference(batch_token, dataset_name, dataset_size, model_name, model_args, img_proc_args):
    with tqdm(total=dataset_size, desc="推理进度", ncols=80) as bar:
        def each_img_finish_callback(img, result):
            bar.update(1)

        inference_detector_4_img_batch(batch_token, model_name, model_args, img_proc_args,
                                    dataset_name, dataset_size, None, each_img_finish_callback)
