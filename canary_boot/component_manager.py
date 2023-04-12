# 管理器
from canary_sefi.core.component.component_manager import SEFI_component_manager
# Canary Lib
from canary_lib import canary_lib

# 数据集
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset  # IMAGENET2012
from Dataset.CIFAR10.dataset_loader import sefi_component as cifar10_dataset  # CIFAR10
from Dataset.MNIST.dataset_loader import sefi_component as mnist_dataset  # F-MNIST

# 防御
from Defense_Method.Adversarial_Training.trades import sefi_component as trades_component

# 图片预处理
from Defense_Method.Img_Preprocess.quantize_trans import sefi_component as quantize_component
from Defense_Method.Img_Preprocess.jpeg_trans import sefi_component as jpeg_component
from Defense_Method.Img_Preprocess.tvm.tvm_trans import sefi_component as tvm_component

def init_component_manager():
    dataset_list = [
        imgnet2012_dataset, cifar10_dataset, mnist_dataset
    ]
    defense_list = [
        trades_component
    ]
    trans_list = [
        quantize_component, jpeg_component, tvm_component
    ]
    SEFI_component_manager.add_all(dataset_list + defense_list + trans_list)
    SEFI_component_manager.add_all(canary_lib)
