from distutils.util import strtobool

from flask import Blueprint, request

from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.entity.msg_entity import MsgEntity


api = Blueprint('task_status_api', __name__)


@api.route('/task/runningStatus', methods=['GET'])
def running_status():
    batch_token = request.args.get("batchToken")
    task = task_thread.execute_task_list.get(batch_token, None)
    if task is None or task.done():
        if check_abnormal_termination(batch_token):
            return MsgEntity("SUCCESS", "1", "finished").msg2json()
        else:
            return MsgEntity("SUCCESS", "-1", "unfinished").msg2json()
    else:
        if task_thread.stop_flag:
            return MsgEntity("SUCCESS", "3", "stopping").msg2json()
        return MsgEntity("SUCCESS", "2", "running").msg2json()

def check_abnormal_termination(batch_token):
    progress_log = global_system_log.get_all_task_progress_log(batch_token)
    for log in progress_log:
        if not strtobool(log.get("is_finish")):
            return False
    else:
        return True


@api.route('/task/stopTask', methods=['GET'])
def stop_task():
    batch_token = request.args.get("batchToken")
    task_thread.stop_task(batch_token)
    return MsgEntity("SUCCESS", "1", "stopping").msg2json()


@api.route('/task/getTaskStepLog', methods=['GET'])
def get_task_progress_log():
    batch_token = request.args.get("batchToken")
    log = global_system_log.get_all_task_progress_log(batch_token)
    return MsgEntity("SUCCESS", "1", log).msg2json()


@api.route('/task/getTaskConsoleLog', methods=['GET'])
def get_task_console_log():
    batch_token = request.args.get("batchToken")
    before_time = request.args.get("beforeTime", None)
    log = global_system_log.get_all_console_msg(batch_token, before_time)
    return MsgEntity("SUCCESS", "1", log).msg2json()
