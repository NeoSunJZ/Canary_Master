import importlib


def load_test_config(attack_config, data_config, model_config, attack_batch_config=None, model_batch_config=None):
    config = dict()
    params_attack = importlib.import_module('defaultConfig.attack.'+attack_config)
    params_data = importlib.import_module('defaultConfig.data.' + data_config)
    params_model = importlib.import_module('defaultConfig.model.' + model_config)
    config.update(params_data.config)
    model_list = params_model.object_list
    config.update({"model_list": model_list})
    attacker_list = dict()
    for attack in params_attack.object_list:
        attacker_list.update({
            attack: model_list
        })
    config.update({"attacker_list": attacker_list})
    config.update(params_attack.config)
    config.update(params_model.config)
    if attack_batch_config is not None:
        params_attack_batch = importlib.import_module('defaultConfig.batch.attack.' + attack_batch_config)
        config.update(params_attack_batch.config)
    if model_batch_config is not None:
        params_model_batch = importlib.import_module('defaultConfig.batch.model.' + model_batch_config)
        config.update(params_model_batch.config)
    print("Loaded configuration details: {}".format(config))
    return config