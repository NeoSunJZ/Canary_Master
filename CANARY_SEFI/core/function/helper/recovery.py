from colorama import Fore

from CANARY_SEFI.batch_manager import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.system_log import global_system_log


class Recovery:
    def __init__(self):
        self.is_recovery_mode = False

    def start_recovery_mode(self, batch_id):
        self.is_recovery_mode = True
        batch_flag.set_batch(batch_id)

    def check_skip(self, participant):
        if self.is_recovery_mode is not True:
            global_system_log.new_record(participant)
            return False, 0
        log = global_system_log.get_current_step_progress_log(participant)
        if log is None:
            global_system_log.new_record(participant)
            return False, 0
        elif bool(int(log.get("is_finish", 0))):
            reporter.console_log("[RECOVERY] 步骤已完成，跳过执行", Fore.GREEN, type="SUCCESS")
            return True, None
        else:
            global_system_log.system_log_id = log.get('id')
            global_system_log.update_finish_status(is_restart = True)
            return False, log.get("completed_num")


global_recovery = Recovery()
