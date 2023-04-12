# 模型(CIFAR-100)
from .MobileNetV2.MobileNetV2_CIFAR100 import sefi_component as mobilenet_model_cifar100
from .ResNet.ResNet_CIFAR100 import sefi_component as resnet_model_cifar100
from .common import sefi_component as common_cifar100

model_list = [
    mobilenet_model_cifar100,
    resnet_model_cifar100,
    common_cifar100
]