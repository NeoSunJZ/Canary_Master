from flask import Blueprint, request

from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.entity.msg_entity import MsgEntity


api = Blueprint('task_status_api', __name__)


@api.route('/task/runningStatus', methods=['GET'])
def running_status():
    task_id = request.args.get("taskID")
    task = task_thread.execute_task_list.get(task_id, None)
    if task is None:
        return MsgEntity("SUCCESS", "0", "no task").msg2json()
    if task.done():
        return MsgEntity("SUCCESS", "1", "finished").msg2json()
    else:
        if task_thread.stop_flag:
            return MsgEntity("SUCCESS", "3", "stopping").msg2json()
        return MsgEntity("SUCCESS", "2", "running").msg2json()


@api.route('/task/stopTask', methods=['GET'])
def stop_task():
    task_id = request.args.get("taskID")
    task_thread.stop_task(task_id)
    return MsgEntity("SUCCESS", "1", "stopping").msg2json()


@api.route('/task/getTaskStepLog', methods=['GET'])
def get_task_progress_log():
    batch_id = request.args.get("batchID")
    log = global_system_log.get_all_task_progress_log(batch_id)
    return MsgEntity("SUCCESS", "1", log).msg2json()


@api.route('/task/getTaskConsoleLog', methods=['GET'])
def get_task_console_log():
    batch_id = request.args.get("batchID")
    before_time = request.args.get("beforeTime", None)
    log = global_system_log.get_all_console_msg(batch_id, before_time)
    return MsgEntity("SUCCESS", "1", log).msg2json()
