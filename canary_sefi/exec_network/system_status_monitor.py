import os
import random
import string
import time

import psutil
import pynvml
import platform

from flask import Blueprint, request

from canary_sefi.entity.msg_entity import MsgEntity
from flask import Blueprint, current_app

from canary_lib.copyright import get_system_version as get_system_sefi_lib_version
from canary_sefi.copyright import get_system_version as get_system_sefi_version
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.core.function.helper.task_thread import monitor_thread
from canary_sefi.entity.msg_entity import MsgEntity
from canary_sefi.task_manager import task_manager

api = Blueprint('system_status_monitor_api', __name__)


@api.route('/info/getSystemInfo', methods=['GET'])
def get_system_info():
    pynvml.nvmlInit()
    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
    system_info = {
        "deviceName": platform.node(),
        "osName": platform.system(),
        "osVersion": platform.version(),
        "cpuName": get_cpu_name(),
        "gpuName": pynvml.nvmlDeviceGetName(gpu_device),
        "gpuMemorySize": "%.2f" % (pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total / (1024 * 1024 * 1024)),
        "memorySize": "%.2f" % (psutil.virtual_memory().total / (1024 * 1024 * 1024)),
    }
    return MsgEntity("SUCCESS", "1", system_info).msg2json()


def get_cpu_name():
    if platform.system() == 'Linux':
        cpu_info = os.popen('cat /proc/cpuinfo |grep "model name" -m 1|awk -F: "{print $2}"').read()
        print(cpu_info)
        cpu_name = cpu_info.strip().replace('model name	: ', '')
    elif platform.system() == 'Windows':
        cpu_info = os.popen('wmic cpu get Name').read()
        cpu_name = cpu_info.replace('\n', '').replace('Name', '').replace('  ', '')
    else:
        cpu_name = "Unknown"
    return cpu_name


def get_system_usage(context):
    with context():
        while True:
            try:
                pynvml.nvmlInit()
                gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
                system_usage = {
                    "cpuUsage": psutil.cpu_percent(),
                    "gpuUsage": pynvml.nvmlDeviceGetUtilizationRates(gpu_device).gpu,
                    "cpuUseMemorySize": "%.1f" % (psutil.virtual_memory().used / (1024 * 1024 * 1024)),
                    "gpuUseMemorySize": "%.1f" % (pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used / (1024 * 1024 * 1024))
                }
                reporter.send_realtime_msg(msg=system_usage, type="USAGE")
                time.sleep(1)
                monitor_thread.watch_dog -= 1
                if monitor_thread.stop_flag or monitor_thread.watch_dog <= 0:
                    monitor_thread.stop_flag = False
                    break
            except Exception as e:
                print(e)


@api.route('/info/startSystemMonitor', methods=['GET'])
def start_system_monitor():
    if task_manager.monitor_token is not None:
        end_system_monitor()
    task_manager.monitor_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    print("starttoken:" + task_manager.monitor_token)
    monitor_thread.execute_task(task_manager.monitor_token, get_system_usage, current_app.app_context)
    print("start-----------monitor")
    print("starttoken:" + task_manager.monitor_token)
    return MsgEntity("SUCCESS", "1", task_manager.monitor_token).msg2json()


@api.route('/info/endSystemMonitor', methods=['GET'])
def end_system_monitor():
    print("stoptoken:" + task_manager.monitor_token)
    monitor_thread.stop_task(task_manager.monitor_token)
    time.sleep(1)
    print("stop-----------monitor")
    print("stoptoken:" + task_manager.monitor_token)
    task_manager.monitor_token = None
    return MsgEntity("SUCCESS", "1", None).msg2json()


@api.route('/info/refreshSystemMonitorWatchDog', methods=['GET'])
def refresh_system_monitor_watch_dog():
    monitor_thread.watch_dog = 10
    return MsgEntity("SUCCESS", "1", None).msg2json()


@api.route('/info/getSystemVersion', methods=['GET'])
def get_system_version():
    print("get_system_version------------------------------")
    system_version = {
        "sefiVersion": get_system_sefi_version(),
        "sefiLibVersion": get_system_sefi_lib_version()
    }
    return MsgEntity("SUCCESS", "1", system_version).msg2json()
