# 模型(FashionMNIST)
from .ResNet.ResNet_FashionMNIST import sefi_component as resnet_model_f_mnist
from .AlexNet.AlexNet_FashionMNIST import sefi_component as alexnet_model_f_mnist
from .LeNetV5.LeNetV5_FashionMNIST import sefi_component as lenetv5_model_f_mnist
from .common import sefi_component as common_f_mnist

model_list = [
    resnet_model_f_mnist,
    alexnet_model_f_mnist,
    lenetv5_model_f_mnist,
    common_f_mnist
]