from CANARY_SEFI.evaluator.logger.log import log


class AdvExampleDALogger:
    def __init__(self, adv_name):
        self.adv_name = adv_name
        self.maximum_disturbance = None
        self.euclidean_distortion = None
        self.pixel_change_ratio = None
        self.deep_metrics_similarity = None
        self.low_level_metrics_similarity = None

    def next(self, batch_token):
        log.adv_disturbance_aware_data["atk_adv_name"].append(self.adv_name)
        log.adv_disturbance_aware_data["AEs_maximum_disturbance"].append(self.maximum_disturbance)
        log.adv_disturbance_aware_data["AEs_euclidean_distortion"].append(self.euclidean_distortion)
        log.adv_disturbance_aware_data["AEs_pixel_change_ratio"].append(self.pixel_change_ratio)
        log.adv_disturbance_aware_data["AEs_deep_metrics_similarity"].append(self.deep_metrics_similarity)
        log.adv_disturbance_aware_data["AEs_low_level_metrics_similarity"].append(self.low_level_metrics_similarity)

        print('\n')
        print('-->[ SEFI 日志记录 ] 对抗样本 {} 扰动感知评估： MD(L-inf) {}, ED(L2) {}, PCR(L0) {}, DMS(DISTS) {}, LMS(MS-GMSD) {}'.format(
            self.adv_name, self.maximum_disturbance, self.euclidean_distortion, self.pixel_change_ratio,
            self.deep_metrics_similarity, self.low_level_metrics_similarity))
        log.save_adv_da_log(batch_token)
        self.adv_name = None
        self.deep_metrics_similarity = None
