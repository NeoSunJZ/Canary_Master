# 管理器
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
# 模型(ImageNet)
from Model.ImageNet.Alexnet_ImageNet import sefi_component as alexnet_model_imagenet  # Alexnet
from Model.ImageNet.ConvNeXt_ImageNet import sefi_component as convnext_model_imagenet  # ConvNext
from Model.ImageNet.DenseNet_ImageNet import sefi_component as densenet_model_imagenet  # DenseNet
from Model.ImageNet.EfficientNet_ImageNet import sefi_component as efficientnet_model_imagenet  # EfficientNet
from Model.ImageNet.EfficientNetV2_ImageNet import sefi_component as efficientnetV2_model_imagenet  # EfficientNetV2
from Model.ImageNet.GoogLeNet_ImageNet import sefi_component as googlenet_model_imagenet  # GoogLeNet
from Model.ImageNet.InceptionV3_ImageNet import sefi_component as inceptionV3_model_imagenet  # InceptionV3
from Model.ImageNet.MNASNet_ImageNet import sefi_component as mnasnet_model_imagenet  # MNASNet
from Model.ImageNet.MobileNetV2_ImageNet import sefi_component as mobilenetv2_model_imagenet  # MobileNetV2
from Model.ImageNet.MobileNetV3_ImageNet import sefi_component as mobilenetv3_model_imagenet  # MobileNetV3
from Model.ImageNet.RegNet_ImageNet import sefi_component as regnet_model_imagenet  # RegNet
from Model.ImageNet.ResNet_ImageNet import sefi_component as resnet_model_imagenet  # ResNet
from Model.ImageNet.ResNeXt_ImageNet import sefi_component as resnext_model_imagenet  # ResNeXt
from Model.ImageNet.ShuffleNetV2_ImageNet import sefi_component as shufflenetV2_model_imagenet  # ShuffleNetV2
from Model.ImageNet.SqueezeNet_ImageNet import sefi_component as squeezenet_model_imagenet  # SqueezeNet
from Model.ImageNet.SwinTransformer_ImageNet import sefi_component as swintransformer_model_imagenet  # SwinTransformer
from Model.ImageNet.VGG_ImageNet import sefi_component as vgg_model_imagenet  # VGG
from Model.ImageNet.VisionTransformer_ImageNet import sefi_component as vit_model_imagenet  # ViT
from Model.ImageNet.WideResNet_ImageNet import sefi_component as wideresnet_model_imagenet  # WideResNet
from Model.ImageNet.common import sefi_component as common_imagenet
# 模型(CIFAR-10)
from Model.CIFAR10.DenseNet.DenseNet_CIFAR10 import sefi_component as densenet_model_cifar10
from Model.CIFAR10.common import sefi_component as common_cifar10
# 攻击方案
from Attack_Method.white_box_adv.CW import sefi_component as cw_attacker  # CW
from Attack_Method.white_box_adv.FGM import sefi_component as fgm_attacker  # FGM
from Attack_Method.white_box_adv.I_FGSM import sefi_component as i_fgsm_attacker  # I-FGSM
from Attack_Method.white_box_adv.MI_FGSM import sefi_component as mi_fgsm_attacker  # MI-FGSM
from Attack_Method.white_box_adv.PGD import sefi_component as pgd_attacker  # PGD
from Attack_Method.white_box_adv.UAP import sefi_component as uap_attacker  # UAP
from Attack_Method.white_box_adv.EAD import sefi_component as ead_attacker  # EAD
from Attack_Method.white_box_adv.DeepFool import sefi_component as deepfool_attacker  # DeepFool
from Attack_Method.black_box_adv.boundary_attack.boundary_attack import sefi_component as boundary_attacker  # BA
from Attack_Method.black_box_adv.gen_attack.gen_attack import sefi_component as gen_attacker  # GA
from Attack_Method.black_box_adv.hop_skip_jump_attack.hop_skip_jump_attack import sefi_component as hsj_attacker  # HSJA
from Attack_Method.black_box_adv.local_search_attack import sefi_component as ls_attacker  # LSA
# 数据集
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset  # IMAGENET2012
from Dataset.CIFAR10.dataset_loader import sefi_component as cifar10_dataset  # CIFAR10


def init_component_manager():
    imagenet_model_list = [
        alexnet_model_imagenet, convnext_model_imagenet, densenet_model_imagenet, efficientnet_model_imagenet,
        efficientnetV2_model_imagenet, googlenet_model_imagenet, inceptionV3_model_imagenet, mnasnet_model_imagenet,
        mobilenetv2_model_imagenet, mobilenetv3_model_imagenet, regnet_model_imagenet, resnet_model_imagenet,
        resnext_model_imagenet, shufflenetV2_model_imagenet, squeezenet_model_imagenet, swintransformer_model_imagenet,
        vgg_model_imagenet, vit_model_imagenet, wideresnet_model_imagenet,
        common_imagenet
    ]
    cifar_model_list = [
        densenet_model_cifar10,
        common_cifar10
    ]

    attacker_list = [
        cw_attacker, fgm_attacker, mi_fgsm_attacker, uap_attacker, deepfool_attacker, boundary_attacker, hsj_attacker,
        pgd_attacker, ls_attacker, ead_attacker, gen_attacker, i_fgsm_attacker
    ]

    dataset_list = [
        imgnet2012_dataset, cifar10_dataset
    ]

    SEFI_component_manager.add_all(imagenet_model_list + cifar_model_list + attacker_list + dataset_list)
