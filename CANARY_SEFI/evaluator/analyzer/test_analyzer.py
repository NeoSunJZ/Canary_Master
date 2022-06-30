import json

import numpy as np

from CANARY_SEFI.handler.csv_handler.csv_io_handler import get_log_data_to_file
from sklearn.metrics import f1_score, accuracy_score


class TestAnalyzer:
    def __init__(self):
        self.analysis_atk_data = {
            "atk_name": [],
            "atk_base_model_name": [],
            "atk_adv_name": [],

            "misclassification_ratio": [],
            "average_increase_adversarial_class_confidence": [],
            "average_reduction_true_class_confidence": [],
            "average_cost_time": [],

            "average_AEs_maximum_disturbance": [],
            "average_AEs_euclidean_distortion": [],
            "average_AEs_pixel_change_ratio": [],
            "average_AEs_deep_metrics_similarity": [],
            "average_AEs_low_level_metrics_similarity": [],
        }
        self.analysis_model_data = {
            "model_name": [],
            "clear_acc": [],
            "clear_F1": [],
            "clear_conf": [],
        }

    def test_result_analysis(self, batch_token, type):
        if type == "model":
            pd_inference_log = get_log_data_to_file("inference_log_" + batch_token + ".csv")
            ori_labels = pd_inference_log.loc[:, 'ori_label'].values
            inference_labels = pd_inference_log.loc[:, 'inference_label'].values

            self.analysis_model_data["clear_acc"].append(accuracy_score(ori_labels, inference_labels))
            self.analysis_model_data["clear_F1"].append(f1_score(ori_labels, inference_labels, average='macro'))
            self.analysis_model_data["clear_conf"].append(calc_conf(pd_inference_log))

            print(self.analysis_model_data)
        else:
            pd_attack_log = get_log_data_to_file("attack_log_" + batch_token + ".csv")
            pd_adv_da_log = get_log_data_to_file("adv_da_log_" + batch_token + ".csv")
            pd_inference_adv_log = get_log_data_to_file("inference_log_" + batch_token + "_ADV.csv")


def calc_conf(pd_inference_log):
    count = 0
    total_conf = 0.0
    for index, row in pd_inference_log.iterrows():
        count += 1
        inference_conf_array = np.array(eval(''.join(row["inference_conf_array"].replace(' ', ' ').replace(' ', ', '))))
        total_conf += inference_conf_array[row["ori_label"]]
    conf = total_conf / count
    return conf
