#!/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask

# from attacker_controller import api as attacker_api
# from dataset_controller import api as dataset_api
# from model_controller import api as model_api
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.exec_network.security_evaluation_controller import api as security_evaluation_api
from canary_sefi.exec_network.task_controller import api as task_api
from canary_sefi.exec_network.analyzer_controller import api as analyzer_api
from canary_sefi.exec_network.client_declaration import client_declaration
from canary_sefi.exec_network.system_status_monitor import api as system_status_monitor_api
from canary_sefi.exec_network.check_code import api as check_code_api
from canary_sefi.exec_network.fast_test.fast_test_controller import api as fast_test_api

# 创建app
def create_app():
    app = Flask("Canary_SEFI")
    app.debug = True

    # 注册蓝图
    app.register_blueprint(client_declaration)
    app.register_blueprint(security_evaluation_api, url_prefix='/core/')
    app.register_blueprint(task_api, url_prefix='/helper/')
    app.register_blueprint(analyzer_api, url_prefix='/analyzer/')
    app.register_blueprint(system_status_monitor_api, url_prefix='/system/')
    app.register_blueprint(check_code_api, url_prefix='/code/')
    app.register_blueprint(fast_test_api, url_prefix='/fastTest/')
    return app
