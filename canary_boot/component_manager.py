# 管理器
from canary_sefi.core.component.component_manager import SEFI_component_manager
# Canary Lib
from canary_lib import canary_lib

# 数据集
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset  # IMAGENET2012
from Dataset.CIFAR10.dataset_loader import sefi_component as cifar10_dataset  # CIFAR10
from Dataset.CIFAR100.dataset_loader import sefi_component as cifar100_dataset  # CIFAR10
from Dataset.MNIST.dataset_loader import sefi_component as mnist_dataset  # F-MNIST


def init_component_manager():
    dataset_list = [
        imgnet2012_dataset, cifar10_dataset, mnist_dataset, cifar100_dataset
    ]
    SEFI_component_manager.add_all(dataset_list)
    SEFI_component_manager.add_all(canary_lib)
