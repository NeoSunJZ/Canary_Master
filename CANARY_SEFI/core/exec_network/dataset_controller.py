import base64
import json
import os

from flask import Blueprint, request, Response

from CANARY_SEFI.core.config.config_manager import config_manager

api = Blueprint('dataset_api', __name__)
config = config_manager.config

@api.route('/dataset/getFileList', methods=['GET'])
def get_file_list():
    dataset_path = config.get("dataset", {}).get(request.args.get("datasetName"), {}).get("path", None)
    page_num = request.args.get("pageNum")
    page_size = request.args.get("pageSize")
    index = 0
    file_list = []
    for files in os.listdir(dataset_path):
        if int(page_num) * int(page_size) > index >= int(page_size) * (int(page_num) - 1):
            with open(dataset_path + files, 'rb') as f:
                file_list.append({
                    'name': files,
                    'data': 'data:image/jpg;base64,' + base64.b64encode(f.read()).decode("utf-8")
                })
        index = index + 1

    return Response(json.dumps(file_list, ensure_ascii=False), mimetype='application/json')
