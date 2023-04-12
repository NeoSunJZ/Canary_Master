# 管理器
from canary_sefi.core.component.component_manager import SEFI_component_manager
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
# 模型(API)
from Model.RemoteAPI.BAIDU import sefi_component as baidu_api
from Model.RemoteAPI.ALIBABA import sefi_component as alibaba_api
from Model.RemoteAPI.HUAWEI import sefi_component as huawei_api
from Model.RemoteAPI.TENCENT import sefi_component as tencent_api
# 攻击方案
from Attack_Method.white_box_adv.l_bfgs.l_bfgs import sefi_component as l_bfgs_attacker  # CW
from Attack_Method.white_box_adv.cw.cw import sefi_component as cw_attacker  # CW
from Attack_Method.white_box_adv.fgsm_family.fgm import sefi_component as fgm_attacker  # FGM
from Attack_Method.white_box_adv.fgsm_family.fgsm import sefi_component as fgsm_attacker  # FGSM
from Attack_Method.white_box_adv.fgsm_family.i_fgsm import sefi_component as i_fgsm_attacker  # I-FGSM
from Attack_Method.white_box_adv.fgsm_family.pgd import sefi_component as pgd_attacker  # PGD
from Attack_Method.white_box_adv.fgsm_family.mi_fgsm import sefi_component as mi_fgsm_attacker  # MI-FGSM
from Attack_Method.white_box_adv.fgsm_family.vmi_fgsm import sefi_component as v_mi_fgsm_attacker  # VMI-FGSM
from Attack_Method.white_box_adv.fgsm_family.ni_fgsm import sefi_component as ni_fgsm_attacker  # PNA-NI-FGSM
from Attack_Method.white_box_adv.fgsm_family.si_fgsm import sefi_component as si_fgsm_attacker  # PNA-SI-FGSM
from Attack_Method.white_box_adv.jsma.jsma import sefi_component as jsma_attacker  # JSMA
from Attack_Method.white_box_adv.UAP import sefi_component as uap_attacker  # UAP
from Attack_Method.white_box_adv.deepfool.deepfool import sefi_component as deepfool_attacker  # DeepFool
from Attack_Method.white_box_adv.ead.EAD import sefi_component as ead_attacker
from Attack_Method.white_box_adv.ssah.ssah import sefi_component as ssah_attacker
from Attack_Method.black_box_adv.adv_gan.adv_gan import sefi_component as adv_gan_attacker  # advGan
from Attack_Method.black_box_adv.boundary_attack.boundary_attack import sefi_component as boundary_attacker  # BA
from Attack_Method.black_box_adv.gen_attack.gen_attack import sefi_component as gen_attacker  # GA
from Attack_Method.black_box_adv.hop_skip_jump_attack.hop_skip_jump_attack import sefi_component as hsj_attacker  # HSJA
from Attack_Method.black_box_adv.local_search_attack.local_search_attack import sefi_component as ls_attacker  # LSA
from Attack_Method.black_box_adv.spsa.spsa import sefi_component as sps_attacker  # SPSA

# 防御
from Defense_Method.Adversarial_Training.trades import sefi_component as trades_component

# 图片预处理
from Defense_Method.Img_Preprocess.quantize_trans import sefi_component as quantize_component
from Defense_Method.Img_Preprocess.jpeg_trans import sefi_component as jpeg_component
from Defense_Method.Img_Preprocess.tvm.tvm_trans import sefi_component as tvm_component
from Attack_Method.black_box_adv.qFool import sefi_component as qfool_attacker  # qFool
from Attack_Method.black_box_adv.tremba.tremba import sefi_component as tremba_attacker # tremba

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
    api_model_list = [
        baidu_api, alibaba_api, huawei_api, tencent_api
    ]

    attacker_list = [
        cw_attacker, fgm_attacker, mi_fgsm_attacker, uap_attacker, deepfool_attacker, boundary_attacker, hsj_attacker,
        pgd_attacker, ls_attacker, ead_attacker, gen_attacker, i_fgsm_attacker, fgsm_attacker, jsma_attacker,
        sps_attacker, l_bfgs_attacker, adv_gan_attacker, v_mi_fgsm_attacker, ni_fgsm_attacker, si_fgsm_attacker,
        qfool_attacker, ssah_attacker, tremba_attacker
    ]

    dataset_list = [
        imgnet2012_dataset, cifar10_dataset
    ]

    defense_list = [
        trades_component
    ]

    trans_list = [
        quantize_component, jpeg_component, tvm_component
    ]

    SEFI_component_manager.add_all(imagenet_model_list + cifar_model_list + api_model_list + attacker_list + dataset_list + defense_list + trans_list)
