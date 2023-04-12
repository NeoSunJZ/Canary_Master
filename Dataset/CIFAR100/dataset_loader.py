import random
import torch
import torchvision
from numpy import long
from torch.utils.data import DataLoader

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.util(util_type=SubComponentType.DATASET_LOADER, util_target=ComponentType.DATASET, name="CIFAR-100")
def dataset_getter(dataset_path, dataset_seed, dataset_size=None):
    dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True)
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