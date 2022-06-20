from CANARY_SEFI.evaluator.logger.log import log


class AdvAttackLogger:
    def __init__(self, atk_name, model_name):
        self.atk_name = str(atk_name)
        self.model_name = str(model_name)
        self.adv_name = None
        self.cost_time = None
        self.ori_label = None

    def next(self, batch_token):
        log.attack_log_data["atk_name"].append(self.atk_name)
        log.attack_log_data["atk_base_model_name"].append(self.model_name)
        log.attack_log_data["atk_adv_name"].append(self.adv_name)
        log.attack_log_data["atk_cost_time"].append(self.cost_time)
        log.attack_log_data["ori_label"].append(self.ori_label)
        print('\n')
        print('-->[ SEFI 日志记录 ] 方法 {} (基于 {}) 生成对抗样本 {}, 耗时{}'.format(
            str(self.atk_name), str(self.model_name), self.adv_name, self.cost_time))
        log.save_attack_log(batch_token)
        self.adv_name = None
        self.cost_time = None
