import json


class ConfigManager:
    def __init__(self):
        self.config = json.load(open("config.json", encoding='utf-8'))


config_manager = ConfigManager()
