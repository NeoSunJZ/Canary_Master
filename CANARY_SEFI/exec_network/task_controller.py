from distutils.util import strtobool

from flask import Blueprint, request

from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.entity.msg_entity import MsgEntity


api = Blueprint('task_status_api', __name__)


@api.route('/task/runningStatus', methods=['GET'])
def running_status():
    task_token = request.args.get("batchToken")
    task = task_thread.execute_task_list.get(task_token, None)
    if task is None or task.done():
        if check_abnormal_termination(task_token):
            return MsgEntity("SUCCESS", "1", "finished").msg2json()
        else:
            return MsgEntity("SUCCESS", "-1", "unfinished").msg2json()
    else:
        if task_thread.stop_flag:
            return MsgEntity("SUCCESS", "3", "stopping").msg2json()
        return MsgEntity("SUCCESS", "2", "running").msg2json()


def check_abnormal_termination(task_token):
    # 初始化批次
    try:
        task_manager.load_task(task_token)
    except FileNotFoundError:
        return False

    progress_log = task_manager.sys_log_logger.get_all_task_progress_log()
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
    # 初始化批次
    task_manager.load_task(request.args.get("batchToken"))

    log = task_manager.sys_log_logger.get_all_task_progress_log()
    return MsgEntity("SUCCESS", "1", log).msg2json()


@api.route('/task/revokeTaskStepLog', methods=['GET'])
def revoke_task_progress_log():
    # 初始化批次
    task_manager.load_task(request.args.get("batchToken"))

    task_manager.sys_log_logger.system_log_id = request.args.get("systemLogID")
    log = task_manager.sys_log_logger.update_finish_status(is_finish=False, stop_reason=None, is_restart=True)

    return MsgEntity("SUCCESS", "1", log).msg2json()


@api.route('/task/getTaskConsoleLog', methods=['GET'])
def get_task_console_log():
    # 初始化批次
    task_manager.load_task(request.args.get("batchToken"))
    before_time = request.args.get("beforeTime", None)
    log = task_manager.sys_log_logger.get_all_console_msg(before_time)
    return MsgEntity("SUCCESS", "1", log).msg2json()
