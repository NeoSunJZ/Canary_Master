import json
import sys

from colorama import Fore

from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.system_log import global_system_log
import traceback

def excepthook(exc_type, exc_value, exc_traceback):
    stop_reason = {
        "exception_type":str(exc_type),
        "exception_object":str(exc_value),
        "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    }

    reporter.console_log(json.dumps(stop_reason), Fore.RED, type="ERROR")
    global_system_log.update_finish_status(False, "SYSTEM_ERROR:" + str(json.dumps(stop_reason)))


sys.excepthook = excepthook
