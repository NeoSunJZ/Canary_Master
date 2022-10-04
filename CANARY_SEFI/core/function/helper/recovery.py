from colorama import Fore

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


class Recovery:
    def __init__(self):
        self.is_recovery_mode = False

    def start_recovery_mode(self, batch_token):
        self.is_recovery_mode = True
        batch_manager.init_batch(batch_token)

    def check_skip(self, participant):
        if self.is_recovery_mode is not True:
            batch_manager.sys_log_logger.new_record(participant)
            return False, 0
        log = batch_manager.sys_log_logger.get_current_step_progress_log(participant)
        if log is None:
            batch_manager.sys_log_logger.new_record(participant)
            return False, 0
        elif bool(int(log.get("is_finish", 0))):
            reporter.console_log("[RECOVERY] 步骤已完成，跳过执行", Fore.GREEN, type="SUCCESS")
            return True, None
        else:
            batch_manager.sys_log_logger.system_log_id = log.get('id')
            batch_manager.sys_log_logger.update_finish_status(is_restart = True)
            return False, log.get("completed_num")


global_recovery = Recovery()
