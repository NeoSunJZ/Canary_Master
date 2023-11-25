import os
import shutil

import torch
from flask import Blueprint, request

from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.component.default_component.dataset_getter import default_folder_dataset_getter
from canary_sefi.entity.dataset_info_entity import DatasetInfo
from canary_sefi.entity.msg_entity import MsgEntity
from canary_sefi.task_manager import task_manager

# 快速测试集成
from .fast_test_model import sefi_component as fast_test_model
SEFI_component_manager.add(fast_test_model)

api = Blueprint('fast_test_api', __name__)


@api.route("/createFastTestTask")
def create_fast_test_task():
    task_token = task_manager.init_task(show_logo=True, is_fast_test=True)
    task_manager.sys_log_logger.new_console_record(msg="Fast Test Task Created!", type="INFO")
    return MsgEntity("SUCCESS", "1", task_token).msg2json()


@api.route("/uploadModel", methods=["POST"])
def upload_model():
    # 加载task
    task_manager.load_task(request.args.get("taskToken"))
    # 获取pth文件
    file = request.files.get("file")

    save_path = task_manager.base_temp_path + 'model/'

    os.makedirs(save_path, exist_ok=True)
    file.save(save_path + file.filename)
    try:
        torch.load(save_path + file.filename)
    except Exception as e:
        return MsgEntity("ERROR", "-1", "[SEFI] Model structure and weight file check error! Info:" + e.__str__()).msg2json()
    return MsgEntity("SUCCESS", "1", {
        "model_list": [
            "FAST-TEST-MODEL"
        ],
        "model_config": {
            "FAST-TEST-MODEL": {
                "model_file_name": file.filename
            }
        },
    }).msg2json()


@api.route("/uploadDataset", methods=["POST"])
def upload_dataset():
    # 加载task
    task_manager.load_task(request.args.get("taskToken"))
    # 获取pth文件
    file = request.files.get("file")

    if ".zip" not in file.filename:
        return MsgEntity("ERROR", "-1", "[SEFI] Dataset file MUST be a Zip file.").msg2json()

    save_path = task_manager.base_temp_path + 'dataset/'
    os.makedirs(save_path, exist_ok=True)
    file.save(save_path + file.filename)

    dataset_folder_name = file.filename.replace(".zip", "")
    os.makedirs(save_path + dataset_folder_name, exist_ok=True)
    try:
        shutil.unpack_archive(save_path + file.filename, save_path + dataset_folder_name, 'zip')
    except Exception as e:
        return MsgEntity("ERROR", "-2", "[SEFI] Dataset file Zip has been broken! Info:" + e.__str__()).msg2json()

    try:
        dataset_info = DatasetInfo("FAST_TEST_DATASET", dataset_extra_info={
            "path": save_path + dataset_folder_name
        })
        dataset = default_folder_dataset_getter(dataset_info)
        if dataset is None:
            return MsgEntity("ERROR", "-3", "[SEFI] The dataset does not meet the requirements! Please make sure to read the user manual!").msg2json()
    except Exception as e:
        return MsgEntity("ERROR", "-3", "[SEFI] The dataset does not meet the requirements! Please make sure to read the user manual!").msg2json()

    return MsgEntity("SUCCESS", "1", {
        "dataset": {
            "dataset_name": "FAST-TEST-DATASET",
            "dataset_folder": dataset_folder_name,
            "is_fast_test": True,
        },
    }).msg2json()
