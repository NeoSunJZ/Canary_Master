#!/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask

# from attacker_controller import api as attacker_api
# from dataset_controller import api as dataset_api
# from model_controller import api as model_api
from CANARY_SEFI.exec_network.security_evaluation_controller import api as security_evaluation_api
from CANARY_SEFI.exec_network.task_controller import api as task_api
from CANARY_SEFI.exec_network.analyzer_controller import api as analyzer_api
from CANARY_SEFI.exec_network.client_declaration import client_declaration


# 创建app
def create_app():
    app = Flask("Canary_SEFI")
    app.debug = True

    # 注册蓝图
    app.register_blueprint(client_declaration)
    app.register_blueprint(security_evaluation_api, url_prefix='/core/')
    app.register_blueprint(task_api, url_prefix='/helper/')
    app.register_blueprint(analyzer_api, url_prefix='/analyzer/')

    return app
