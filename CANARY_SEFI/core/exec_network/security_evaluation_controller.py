import random
import sys
import traceback

from colorama import Fore
from flask import Blueprint, request, current_app

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.excepthook import excepthook
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.core.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.entity.msg_entity import MsgEntity
from CANARY_SEFI.evaluator.logger.db_logger import log

api = Blueprint('test_api', __name__)


def full_test(debug_log, config, context):
    with context():
        try:
            log.debug_log = debug_log
            security_evaluation = SecurityEvaluation()
            security_evaluation.full_security_test(config.get('dataset'), config.get('dataset_size'),
                                                   config.get('dataset_seed',
                                                              random.randint(1000000000000, 10000000000000)),
                                                   config.get('attacker_list'), config.get('attacker_config'),
                                                   config.get('model_config'), config.get('img_proc_config'))
            reporter.console_log("任务已顺利结束", Fore.GREEN, type="SUCCESS", show_batch=True, show_step_sequence=True)
            reporter.send_disconnect()
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            excepthook(exc_type, exc_value, exc_traceback)

@api.route('/test/adv/startFullTest', methods=['POST'])
def start_full_test():
    task_id = request.json.get("taskID")
    batch_flag.new_batch()
    try:
        task_thread.execute_task(task_id, full_test, request.json.get("debugMode", False), request.json.get("config"), current_app.app_context)
    except RuntimeError as e:
        return MsgEntity("ERROR", "-1", e).msg2json()

    return MsgEntity("SUCCESS", "1", batch_flag.batch_id).msg2json()
