import sys

from colorama import Fore
from flask import Blueprint, request, current_app

from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.excepthook import excepthook
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.entity.msg_entity import MsgEntity

api = Blueprint('test_api', __name__)


def full_test(debug_log, config, context):
    with context():
        try:
            task_manager.test_data_logger.debug_log = debug_log
            security_evaluation = SecurityEvaluation(config)
            security_evaluation.attack_full_test(config.get('use_img_file', True),
                                                 config.get('use_raw_nparray_data', False))

            reporter.console_log("任务已顺利结束", Fore.GREEN, type="SUCCESS", show_task=True, show_step_sequence=True)
            reporter.send_disconnect()
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            excepthook(exc_type, exc_value, exc_traceback)


@api.route('/test/adv/startFullTest', methods=['POST'])
def start_full_test():
    restart_flag = request.json.get("isRestart", False)
    if not restart_flag:
        task_manager.init_task(show_logo=True)
        global_recovery.is_recovery_mode = False
    else:
        global_recovery.start_recovery_mode(request.json.get("batchToken"))
        reporter.console_log("恢复测试启动 任务标识{}".format(str(task_manager.task_token)), Fore.GREEN, type="SUCCESS")

    try:
        task_thread.execute_task(task_manager.task_token, full_test, request.json.get("debugMode", False), request.json.get("config"), current_app.app_context)
    except RuntimeError as e:
        return MsgEntity("ERROR", "-1", e).msg2json()

    return MsgEntity("SUCCESS", "1", task_manager.task_token).msg2json()