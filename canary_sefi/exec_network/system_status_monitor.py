import os

import psutil
import pynvml
import platform

from flask import Blueprint, request

from canary_sefi.entity.msg_entity import MsgEntity

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
