import sys

from colorama import Fore
from flask import Blueprint, request, current_app

from CANARY_SEFI.batch_manager import batch_flag
from CANARY_SEFI.core.function.helper.excepthook import excepthook
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.entity.msg_entity import MsgEntity
from CANARY_SEFI.evaluator.logger.test_data_logger import log

api = Blueprint('test_api', __name__)


def full_test(debug_log, config, context):
    with context():
        try:
            log.debug_log = debug_log
            security_evaluation = SecurityEvaluation(config.get('dataset'), config.get('dataset_size'),
                                                     config.get('dataset_seed', None))
            security_evaluation.attack_full_test(config.get('attacker_list'), config.get('attacker_config'),
                                                 config.get('model_list'), config.get('model_config'),
                                                 config.get('img_proc_config'),
                                                 config.get('transfer_attack_test_mode'),
                                                 config.get('transfer_attack_test_on_model_list', None))

            reporter.console_log("任务已顺利结束", Fore.GREEN, type="SUCCESS", show_batch=True, show_step_sequence=True)
            reporter.send_disconnect()
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            excepthook(exc_type, exc_value, exc_traceback)


@api.route('/test/adv/startFullTest', methods=['POST'])
def start_full_test():
    restart_flag = request.json.get("isRestart", False)
    if not restart_flag:
        batch_flag.new_batch()
        global_recovery.is_recovery_mode = False
    else:
        batch_flag.set_batch(request.json.get("batchToken"))
        global_recovery.is_recovery_mode = True
        reporter.console_log("恢复测试启动 任务标识" + str(batch_flag.batch_id), Fore.GREEN, type="SUCCESS")
    try:
        task_thread.execute_task(batch_flag.batch_id, full_test, request.json.get("debugMode", False), request.json.get("config"), current_app.app_context)
    except RuntimeError as e:
        return MsgEntity("ERROR", "-1", e).msg2json()

    return MsgEntity("SUCCESS", "1", batch_flag.batch_id).msg2json()