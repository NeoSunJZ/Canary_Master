#!/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask

from attacker_controller import api as attacker_api
from dataset_controller import api as dataset_api
from model_controller import api as model_api
from client_declaration import client_declaration


# 创建app

def create_app():
    app = Flask(__name__)
    app.debug = True

    # 注册蓝图
    app.register_blueprint(client_declaration)
    app.register_blueprint(attacker_api, url_prefix='/core/')
    app.register_blueprint(model_api, url_prefix='/core/')
    app.register_blueprint(dataset_api, url_prefix='/core/')

    return app
