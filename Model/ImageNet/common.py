import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@sefi_component.util(util_type="img_preprocessor", util_target="model", name="Alexnet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="ConvNext(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="DenseNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="EfficientNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="EfficientNetV2(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="GoogLeNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="InceptionV3(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="MNASNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="MobileNetV2(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="MobileNetV3(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="RegNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="ResNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="ResNeXt(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="ShuffleNetV2(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="SqueezeNet(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="SwinTransformer(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="VGG(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="ViT(ImageNet)")
@sefi_component.util(util_type="img_preprocessor", util_target="model", name="WideResNet(ImageNet)")
def img_pre_handler(img, args):
    img = img.copy().astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="Alexnet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="ConvNext(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="DenseNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="EfficientNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="EfficientNetV2(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="GoogLeNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="InceptionV3(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="MNASNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="MobileNetV2(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="MobileNetV3(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="RegNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="ResNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="ResNeXt(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="ShuffleNetV2(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="SqueezeNet(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="SwinTransformer(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="VGG(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="ViT(ImageNet)")
@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="WideResNet(ImageNet)")
def img_post_handler(adv, args):
    adv = np.squeeze(adv, axis=0)
    adv = adv.transpose(1, 2, 0)
    adv = adv * 255.0
    adv = np.clip(adv, 0, 255).astype(np.float32)
    return adv


@sefi_component.util(util_type="result_postprocessor", util_target="model", name="Alexnet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="ConvNext(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="DenseNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="EfficientNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="EfficientNetV2(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="GoogLeNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="InceptionV3(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="MNASNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="MobileNetV2(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="MobileNetV3(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="RegNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="ResNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="ResNeXt(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="ShuffleNetV2(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="SqueezeNet(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="SwinTransformer(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="VGG(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="ViT(ImageNet)")
@sefi_component.util(util_type="result_postprocessor", util_target="model", name="WideResNet(ImageNet)")
def result_post_handler(result, args):
    probs = F.softmax(result).detach().cpu().numpy()[0]
    pred = np.argmax(probs)
    return pred, probs


@sefi_component.inference_detector(model_name="Alexnet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="ConvNext(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="DenseNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="EfficientNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="EfficientNetV2(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="GoogLeNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="InceptionV3(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="MNASNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="MobileNetV2(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="MobileNetV3(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="RegNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="ResNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="ResNeXt(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="ShuffleNetV2(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="SqueezeNet(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="SwinTransformer(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="VGG(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="ViT(ImageNet)", support_type="numpy_array", return_type="label_string")
@sefi_component.inference_detector(model_name="WideResNet(ImageNet)", support_type="numpy_array", return_type="label_string")
def inference_detector(model, img):
    img_temp = Variable(torch.from_numpy(img).to(device).float())
    model.eval()
    return model(img_temp)
