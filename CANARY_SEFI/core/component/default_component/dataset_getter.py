import random
from typing import Tuple, Any

import numpy as np
import torch
from numpy import long
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from CANARY_SEFI.core.component.component_enum import SubComponentType
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager


def get_dataset(dataset_info):
    dataset_component = SEFI_component_manager.dataset_list.get(dataset_info.dataset_name, default=None,
                                                                allow_not_exist=True)
    if dataset_component is not None:
        dataset_getter = dataset_component.get(SubComponentType.DATASET_LOADER)
    else:
        dataset_getter = default_dataset_getter

    dataset_path = config_manager.config.get("dataset", {}).get(dataset_info.dataset_name, {}).get("path", None)
    if dataset_info.dataset_name == "CIFAR-10":
        dataset = dataset_getter(dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size, dataset_info.is_train)
    else:
        dataset = dataset_getter(dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size)
    return dataset


def default_dataset_getter(dataset_path, dataset_seed, dataset_size=None):

    class ImageFolderCustom(ImageFolder):
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            sample, target = super().__getitem__(index)
            return np.array(sample, dtype=np.uint8), target
    try:
        dataset = ImageFolderCustom(root=dataset_path)
    except Exception as e:
        print(e)
        raise RuntimeError("[SEFI] User has NOT defined dataset loader "
                           "AND default dataset loader is used BUT FAILED TO IMPORT DATASET FROM PATH")

    if dataset_size is not None:
        if dataset_size > len(dataset):
            raise OverflowError("The specified dataset subset size exceeds the dataset itself")

        random.seed(dataset_seed)
        seed = random.randint(1000000000000000, 10000000000000000)
        dataset, __ = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[dataset_size, len(dataset)-dataset_size],
            generator=torch.Generator().manual_seed(long(seed)))
    return dataset
