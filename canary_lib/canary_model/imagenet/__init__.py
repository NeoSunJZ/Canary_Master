# 模型(ImageNet)
from .AlexNet_ImageNet import sefi_component as alexnet_model_imagenet  # Alexnet
from .ConvNeXt_ImageNet import sefi_component as convnext_model_imagenet  # ConvNext
from .DenseNet.DenseNet_ImageNet import sefi_component as densenet_model_imagenet  # DenseNet
from .EfficientNet_ImageNet import sefi_component as efficientnet_model_imagenet  # EfficientNet
from .EfficientNetV2_ImageNet import sefi_component as efficientnetV2_model_imagenet  # EfficientNetV2
from .GoogLeNet_ImageNet import sefi_component as googlenet_model_imagenet  # GoogLeNet
from .InceptionV3_ImageNet import sefi_component as inceptionV3_model_imagenet  # InceptionV3
from .MNASNet_ImageNet import sefi_component as mnasnet_model_imagenet  # MNASNet
from .MobileNetV2_ImageNet import sefi_component as mobilenetv2_model_imagenet  # MobileNetV2
from .MobileNetV3_ImageNet import sefi_component as mobilenetv3_model_imagenet  # MobileNetV3
from .RegNet_ImageNet import sefi_component as regnet_model_imagenet  # RegNet
from .ResNet_ImageNet import sefi_component as resnet_model_imagenet  # ResNet
from .ResNeXt_ImageNet import sefi_component as resnext_model_imagenet  # ResNeXt
from .ShuffleNetV2_ImageNet import sefi_component as shufflenetV2_model_imagenet  # ShuffleNetV2
from .SqueezeNet_ImageNet import sefi_component as squeezenet_model_imagenet  # SqueezeNet
from .SwinTransformer_ImageNet import sefi_component as swintransformer_model_imagenet  # SwinTransformer
from .VGG_ImageNet import sefi_component as vgg_model_imagenet  # VGG
from .VisionTransformer_ImageNet import sefi_component as vit_model_imagenet  # ViT
from .WideResNet_ImageNet import sefi_component as wideresnet_model_imagenet  # WideResNet
from .common import sefi_component as common_imagenet

model_list = [
        alexnet_model_imagenet, convnext_model_imagenet, densenet_model_imagenet, efficientnet_model_imagenet,
        efficientnetV2_model_imagenet, googlenet_model_imagenet, inceptionV3_model_imagenet, mnasnet_model_imagenet,
        mobilenetv2_model_imagenet, mobilenetv3_model_imagenet, regnet_model_imagenet, resnet_model_imagenet,
        resnext_model_imagenet, shufflenetV2_model_imagenet, squeezenet_model_imagenet, swintransformer_model_imagenet,
        vgg_model_imagenet, vit_model_imagenet, wideresnet_model_imagenet,
        common_imagenet
    ]