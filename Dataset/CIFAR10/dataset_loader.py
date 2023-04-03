import random
import torch
import torchvision
from numpy import long
from torch.utils.data import DataLoader

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.util(util_type=SubComponentType.DATASET_LOADER, util_target=ComponentType.DATASET, name="CIFAR-10")
def dataset_getter(dataset_path, dataset_seed, dataset_size=None, is_train=False, val_size=2):
    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=is_train, download=True)
    random.seed(dataset_seed)
    seed = random.randint(1000000000000000, 10000000000000000)

    if dataset_size is not None:
        if dataset_size > len(dataset):
            raise OverflowError("The specified dataset subset size exceeds the dataset itself")
        dataset, __ = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[dataset_size, len(dataset) - dataset_size],
            generator=torch.Generator().manual_seed(long(seed)))

    if is_train:
        train_dataset = dataset
        val_datasets = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True)
        val_dataset, __ = torch.utils.data.random_split(
            dataset=val_datasets,
            lengths=[val_size, len(val_datasets) - val_size],
            generator=torch.Generator().manual_seed(long(seed)))

        return [val_dataset, train_dataset]

    return dataset
