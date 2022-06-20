from CANARY_SEFI.evaluator.logger.log import log


class InferenceLogger:
    def __init__(self, model_name):
        self.model_name = str(model_name)
        self.img_name = None
        self.ori_label = None
        self.inference_conf_array = None
        self.inference_label = None

    def next(self, batch_token):
        log.inference_log_data["model_name"].append(self.model_name)
        log.inference_log_data["img_name"].append(self.img_name)
        log.inference_log_data["ori_label"].append(self.ori_label)
        log.inference_log_data["inference_conf_array"].append(self.inference_conf_array)
        log.inference_log_data["inference_label"].append(self.inference_label)
        print('\n')
        print('-->[ SEFI 日志记录 ] 基于 {} 推理图片 {} 的标签为 {} (数据集标注标签为 {})'.format(
            self.model_name, str(self.img_name), self.inference_label, self.ori_label))
        log.save_inference_log(batch_token)
        self.img_name = None
        self.ori_label = None
        self.inference_conf_array = None
        self.inference_label = None
