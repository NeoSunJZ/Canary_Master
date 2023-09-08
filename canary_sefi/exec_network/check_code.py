from flask import Blueprint, request
from canary_sefi.entity.msg_entity import MsgEntity
from canary_lib.get_attack_file import get_attack_path
import os

api = Blueprint('check_code_api', __name__)


@api.route('/getCode', methods=['GET'])
def get_system_info():
    python_files = [get_files(get_attack_path(request.args.get("attackName")))]
    return MsgEntity("SUCCESS", "1", python_files).msg2json()


def get_files(directory):
    python_files = []
    current_path = os.path.abspath(__file__)
    target_directory = current_path[
                       :current_path.index('canary_sefi')] + "canary_lib\\canary_attack_method\\" + directory
    if not directory.endswith('.py'):
        # 遍历目录中的所有文件和子目录
        for file in os.listdir(target_directory):
            if file.endswith(".py"):  # 仅处理 Python 文件
                file_path = os.path.join(target_directory, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                python_files.append({"filename": file, "content": content})
    else:
        with open(target_directory, 'r', encoding='utf-8') as f:
            content = f.read()
    filename = os.path.split(directory)
    python_files.append({"filename": filename[1], "content": content})
    return python_files
