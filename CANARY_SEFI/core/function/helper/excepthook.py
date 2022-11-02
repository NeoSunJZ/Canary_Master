import json
import sys

from colorama import Fore

from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
import traceback


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exc()
    stop_reason = {
        "exception_type":str(exc_type.__name__),
        "exception_object":str(exc_value),
        "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    }
    try:
        error_msg = json.dumps(stop_reason)
        reporter.console_log(error_msg, Fore.RED, type="ERROR")
        task_manager.sys_log_logger.update_finish_status(False, error_msg)
    except Exception as e:
        traceback.print_exc()