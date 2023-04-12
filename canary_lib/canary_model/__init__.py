# 模型(ImageNet)
from .imagenet import model_list as ml_imagenet
# 模型(CIFAR-10)
from .cifar_10 import model_list as ml_cifar10
# 模型(CIFAR-100)
from .cifar_100 import model_list as ml_cifar100
# 模型(FashionMNIST)
from .fashion_mnist import model_list as ml_fmnist
# 模型(API)
from .remote_api import model_list as ml_api

model_list = ml_imagenet + ml_cifar10 + ml_cifar100 + ml_fmnist + ml_api
