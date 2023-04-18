import json
import os, sys

from colorama import Fore


class ConfigManager:
    def __init__(self, config=None):
        if config is not None:
            self.config = config
            return
        os.chdir(sys.path[0])
        path = os.path.abspath(os.path.join(os.getcwd(), "."))
        if os.path.exists(path + "/config.json"):
            self.config = json.load(open("config.json", encoding='utf-8'))
        else:
            self.config = {
                "appName": "Canary SEFI",
                "appDesc": "Canary SEFI Default Project",
                "datasetPath": "SEFI_temp/dataset/",
                "baseTempPath": "SEFI_temp/data/",
                "centerDatabasePath": "SEFI_temp/data/",
                "system": {
                    "limited_read_img_size": 900,
                    "use_file_memory_cache": True,
                    "save_fig_model": "save_img_file"
                }
            }
            print(Fore.RED + "[SEFI WARNING] Unable to find startup config JSON file, default config used!")
        # 路径检查
        self.config['datasetPath'] = self.path_end_check(self.config['datasetPath'])
        self.config['baseTempPath'] = self.path_end_check(self.config['baseTempPath'])
        self.config['centerDatabasePath'] = self.path_end_check(self.config['centerDatabasePath'])

    @staticmethod
    def path_end_check(path):
        if path[-1] != '/' or path[-1] != '\\':
            return path + '/'
        else:
            return path


config_manager = ConfigManager()
