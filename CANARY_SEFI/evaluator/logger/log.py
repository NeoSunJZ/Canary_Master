from CANARY_SEFI.handler.csv_handler.csv_io_handler import save_log_data_to_file


class Log:
    def __init__(self):
        self.attack_log_data = {}
        self.adv_disturbance_aware_data = {}
        self.inference_log_data = {}

        self.init()

    def init(self):
        self.attack_log_data = {
            "atk_name": [],
            "atk_base_model_name": [],
            "atk_adv_name": [],
            "atk_cost_time": [],
            "ori_label": []
        }
        self.adv_disturbance_aware_data = {
            "atk_adv_name": [],
            "AEs_maximum_disturbance": [],
            "AEs_euclidean_distortion": [],
            "AEs_pixel_change_ratio": [],
            "AEs_deep_metrics_similarity": [],
            "AEs_low_level_metrics_similarity": [],
        }
        self.inference_log_data = {
            "model_name": [],
            "img_name": [],
            "ori_label": [],
            "inference_conf_array": [],
            "inference_label": [],
        }

    def save_attack_log(self, batch_token):
        save_log_data_to_file(self.attack_log_data, "attack_log_" + batch_token + ".csv")

    def save_adv_da_log(self, batch_token):
        save_log_data_to_file(self.adv_disturbance_aware_data, "adv_da_log_" + batch_token + ".csv")

    def save_inference_log(self, batch_token):
        save_log_data_to_file(self.inference_log_data, "inference_log_" + batch_token + ".csv")


log = Log()
