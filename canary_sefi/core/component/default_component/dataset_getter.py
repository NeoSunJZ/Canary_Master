import copy
import importlib
import os
import random
from typing import Tuple, Any
import numpy as np
import torch
from colorama import Fore
from numpy.compat import long
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from canary_sefi.core.component.component_dict import ComponentDict, ComponentDictType
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.entity.dataset_info_entity import DatasetType


def get_dataset(dataset_info):
    dataset_info = copy.copy(dataset_info)
    dataset_component = SEFI_component_manager.dataset_list.get(dataset_info.dataset_name, default=None,
                                                                allow_not_exist=True)
    if dataset_component is not None:
        # 有用户定义的加载器使用用户定义的加载器
        dataset_getter = dataset_component.get(SubComponentType.DATASET_LOADER)
        dataset = dataset_getter(copy.copy(dataset_info))
    else:
        msg = "[SEFI] User has NOT defined dataset loader. Attempting to load dataset from folder..."
        reporter.console_log(msg, Fore.BLUE, show_task=True, show_step_sequence=True)
        dataset = default_folder_dataset_getter(copy.copy(dataset_info))
    if dataset is not None:
        SEFI_component_manager.dataset_list[dataset_info.dataset_name] = ComponentDict({
            SubComponentType.DATASET_LOADER: default_folder_dataset_getter
        },
            dict_type=ComponentDictType.ComponentDict,
            component_type=ComponentType.DATASET)
    else:
        msg = "[SEFI] User has NOT defined dataset loader. Attempting to load dataset by TorchVision's Dataset..."
        reporter.console_log(msg, Fore.BLUE, show_task=True, show_step_sequence=True)
        dataset = default_torchvision_dataset_getter(copy.copy(dataset_info))
        # 检查是否仍然为空
        if dataset is not None:
            SEFI_component_manager.dataset_list[dataset_info.dataset_name] = ComponentDict({
                SubComponentType.DATASET_LOADER: default_torchvision_dataset_getter
            },
                dict_type=ComponentDictType.ComponentDict,
                component_type=ComponentType.DATASET)
        else:
            raise Exception("[SEFI] After exhausting possible loading methods, it has not been successful. "
                            "Please configure the loader or check if the configuration is correct")

    if dataset_info.dataset_size is None:
        dataset_info.dataset_size = len(dataset)
        return dataset, None
    else:
        if dataset_info.dataset_size > len(dataset):
            return dataset, None
        random.seed(dataset_info.dataset_seed)
        seed = random.randint(1000000000000000, 10000000000000000)

        sub_dataset_1, sub_dataset_2 = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[dataset_info.dataset_size, len(dataset) - dataset_info.dataset_size],
            generator=torch.Generator().manual_seed(long(seed)))
        return sub_dataset_1, sub_dataset_2


def default_folder_dataset_getter(dataset_info):
    class ImageFolderCustom(ImageFolder):
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            sample, target = super().__getitem__(index)
            return np.array(sample, dtype=np.uint8), target

    try:
        if not os.path.exists(dataset_info.dataset_path):
            os.makedirs(dataset_info.dataset_path)
        dataset = ImageFolderCustom(root=dataset_info.dataset_path)
        return dataset
    except Exception as e:
        msg = "[SEFI] INFO: User has NOT defined dataset loader " \
              "AND default dataset loader is used BUT FAILED TO IMPORT DATASET FROM PATH\n" \
              "Info: {}".format(e)
        reporter.console_log(msg, Fore.RED, show_task=True, show_step_sequence=True)
        return None


def default_torchvision_dataset_getter(dataset_info):
    try:
        torchvision_dataset = importlib.import_module('torchvision.datasets')
        dataset_class = getattr(torchvision_dataset, dataset_info.dataset_name)
        is_train = dataset_info.dataset_type is not DatasetType.NORMAL_TEST
        dataset = dataset_class(root=dataset_info.dataset_path, train=is_train, download=True)
        return dataset
    except Exception as e:
        msg = "[SEFI] INFO: User has NOT defined dataset loader " \
              "AND default dataset loader is used BUT FAILED TO IMPORT DATASET BY TORCHVISION DATASET\n" \
              "Info: {}".format(e)
        reporter.console_log(msg, Fore.RED, show_task=True, show_step_sequence=True)
        return None
