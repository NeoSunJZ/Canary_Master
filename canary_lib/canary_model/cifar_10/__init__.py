# 模型(CIFAR-10)
from .DenseNet.DenseNet_CIFAR10 import sefi_component as densenet_model_cifar10
from .GoogLeNet.GoogLeNet_CIFAR10 import sefi_component as googlenet_model_cifar10
from .Inception.InceptionV3_CIFAR10 import sefi_component as inception_model_cifar10
from .MobileNet.MobileNetV2_CIFAR10 import sefi_component as mobilenet_model_cifar10
from .ResNet.ResNet_CIFAR10 import sefi_component as resnet_model_cifar10
from .VGG.VGG_CIFAR10 import sefi_component as vgg_model_cifar10
from .common import sefi_component as common_cifar10

model_list = [
    densenet_model_cifar10,
    googlenet_model_cifar10,
    inception_model_cifar10,
    mobilenet_model_cifar10,
    resnet_model_cifar10,
    vgg_model_cifar10,
    common_cifar10
]