import json
import os

from torch.hub import load_state_dict_from_url


def pretrained_weight_loader(weight_path, dataset_name, model_name, model_arch, model, device, progress=True):
    weight_files = json.load(open(os.path.dirname(__file__) + '/weight_files.json', encoding='utf-8'))
    weight_file_info = weight_files.get(dataset_name, {}).get(model_name, {}).get(model_arch, {})

    weight_file_name = weight_path + weight_file_info.get('file_name')
    state_dict = load_state_dict_from_url(
        url=weight_file_info.get('remote_path'),
        model_dir=weight_path,
        map_location=device,
        progress=progress,
        file_name = weight_file_name)
    return model.load_state_dict(state_dict)
