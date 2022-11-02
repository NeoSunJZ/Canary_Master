import numpy
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()

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
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)
        ori_img /= 255.0
        ori_img = ori_img.transpose(2, 0, 1)

        ori_img = Variable(torch.from_numpy(ori_img).to(run_device).float())

        # Resize
        img_size_h = args.get("img_size_h", 224)
        img_size_w = args.get("img_size_w", 224)
        resize = Resize([img_size_h, img_size_w])
        ori_img = resize(ori_img)
        ori_img = torch.unsqueeze(ori_img, dim=0)
        if result is None:
            result = ori_img
        else:
            result = torch.cat((result, ori_img), dim=0)
    return result

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
def img_post_handler(adv_imgs, args):
    if type(adv_imgs) == torch.Tensor:
        adv_imgs = adv_imgs.data.cpu().numpy()

    result = []
    for adv_img in adv_imgs:
        adv_img = adv_img.transpose(1, 2, 0)
        adv_img = adv_img * 255.0
        adv_img = np.clip(adv_img, 0, 255).astype(np.float32)
        result.append(adv_img)
    return result


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
    outputs = torch.nn.functional.softmax(result, dim=1).detach().cpu().numpy()
    predicts = []
    for output in outputs:
        predicts.append(np.argmax(output))
    return predicts, outputs


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
    model.eval()
    return model(img)
