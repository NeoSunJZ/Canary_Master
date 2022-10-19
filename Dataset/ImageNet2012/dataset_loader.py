import random
from typing import Tuple, Any

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from numpy import long
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
sefi_component = SEFIComponent()


class ImageFolderCustom(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = super().__getitem__(index)
        return np.array(sample, dtype=np.uint8), target


@sefi_component.util(util_type="dataset_getter_handler", util_target="dataset", name="ILSVRC-2012")
def dataset_getter(dataset_path, dataset_seed, dataset_size=None):
    dataset = ImageFolderCustom(root=dataset_path)
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

    # img_np_array = np.array(dataset[int(img_index)][0], dtype=np.uint8)
    #
    # if with_label:
    #     return img_np_array, dataset[int(img_index)][1]
    # else:
    #     return img_np_array
