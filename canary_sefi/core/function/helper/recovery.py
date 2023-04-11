from colorama import Fore

from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.helper.realtime_reporter import reporter


class Recovery:
    def __init__(self):
        self.is_recovery_mode = False
        self.force_execute_step_list = None

    def start_recovery_mode(self, task_token, show_logo=False, run_device=None, force_execute_step_list=None):
        self.is_recovery_mode = True
        self.force_execute_step_list = force_execute_step_list
        task_manager.init_task(task_token, show_logo, run_device)

    def check_skip(self, participant):
        if self.is_recovery_mode is not True:
            task_manager.sys_log_logger.new_record(participant)
            return False, 0
        log = task_manager.sys_log_logger.get_current_step_progress_log(participant)
        if log is None:
            task_manager.sys_log_logger.new_record(participant)
            return False, 0
        elif self.force_execute_step_list is not None and log.get("step") in self.force_execute_step_list:
            task_manager.sys_log_logger.system_log_id = log.get('id')
            task_manager.sys_log_logger.update_finish_status(is_restart=True, reset_completed_num=True)
            return False, 0
        elif bool(int(log.get("is_finish", 0))):
            reporter.console_log("[RECOVERY] Step completed, SKIP execution", Fore.RED, type="SUCCESS")
            return True, None
        else:
            task_manager.sys_log_logger.system_log_id = log.get('id')
            task_manager.sys_log_logger.update_finish_status(is_restart=True)
            return False, log.get("completed_num")


global_recovery = Recovery()
