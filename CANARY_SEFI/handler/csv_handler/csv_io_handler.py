import pandas as pd

from CANARY_SEFI.core.config.config_manager import config_manager


def save_log_data_to_file(data, filename):
    df = pd.DataFrame(data)
    temp_path = config_manager.config.get("log", "Log_Temp/")
    df.to_csv(temp_path + filename, index=False)


def get_log_data_to_file(filename):
    temp_path = config_manager.config.get("log", "Log_Temp/")
    return pd.read_csv(temp_path + filename)
