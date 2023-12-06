# Canary SEFI
<img src="https://github.com/NeoSunJZ/Canary_Master/blob/main/logo.png?raw=true" width="200" alt="">

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

[中文版](https://github.com/NeoSunJZ/Canary_Master/blob/main/readme_cn.md)

## Document
**Please refer to our 📖document 👉👉[Canary Document](https://neosunjz.github.io/Canary/)👈👈 for using Canary.**

The document you are currently reading is an early version, which is incomplete and is being edited by the author and will be updated continuously. We will provide the complete document as soon as possible, as well as the English version of the document. If you have questions, please contact `jiazheng.sun@bit.edu.cn` for more information.

## Introduction
SEFI is a framework for evaluating the robustness of deep learning-based image recognition models.

It uses selected attack methods to generate adversarial samples based on selected models and uses these adversarial examples to attack any model you want. In the process, it collects data including multiple evaluation metrics to assess AI model robustness and attack method effectiveness while trying to find the best defense solution.

It also provides a toolkit containing multiple models, SOTA attack methods and defense methods, and allows users to make additional integrations themselves.

SEFI was created and is maintained by researchers at BIT.

## Quick Start

### Install

We provide a stable version on PyPI, which you can install by:

```sh
python -m pip install torch torchvision torchaudio
python -m pip install canary-sefi
```

We recommend that you install PyTorch beforehand. Canary requires at least PyTorch 2.0.1, but we recommend using PyTorch 2.1.1 or higher.

### Example Project

We provide an [Example Project](https://github.com/NeoSunJZ/Canary_Example) located in Github, which you can directly run to try.

Please execute the following command on the terminal to clone the code locally and run the Example Project:

```sh
git clone https://github.com/NeoSunJZ/Canary_Example.git
python run.py
```

## Framework

<img src="https://github.com/NeoSunJZ/Canary_Master/blob/main/framework_structure.png?raw=true" width="800" alt="">

## Functions

### What models do we support？
We have supported 15 models on 4 datasets, of which all 15 models for the ImageNet dataset are available. We hope that you will participate in improving our model library and share your own model structure and weighting information to help more people.

We have built a public repository of model weights(SEFI-LW) located: https://github.com/NeoSunJZ/Canary_SEFI_Lib_Weight

| Models                | Substructure    | Dataset       | Fully Available | Source                                                       | Support GCAM? | Weights Available | Top-1 Acc |
| --------------------- | --------------- | ------------- | --------------- | ------------------------------------------------------------ | ------------- | ----------------- | --------- |
| **LeNetV5**           | N/A             | Fashion-MNIST | Come Soon       | [CNN-for-Fashion-MNIST](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST) | Planned       | ✔SEFI-LW          |           |
| **AlexNet**           | N/A             | Fashion-MNIST | Come Soon       | [CNN-for-Fashion-MNIST](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST) | Planned       | ✔SEFI-LW          | 92.19%    |
| **AlexNet**           | N/A             | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **VGG**               | vgg16_bn        | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **VGG**               | vgg16_bn        | CIFAR-10      | ✔               | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | ✔             | ✔SEFI-LW          |           |
| **VGG**               | vgg16_bn        | CIFAR-100     | ✔               | [PyTorch CIFAR Models](https://github.com/chenyaofo/pytorch-cifar-models) | Planned       | ✔SEFI-LW          |           |
| **GoogLeNet**         | N/A             | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **GoogLeNet**         | N/A             | CIFAR-10      | ✔               | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | ✔             | ✔SEFI-LW          |           |
| **InceptionV3**       | N/A             | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **InceptionV3**       | N/A             | CIFAR-10      | ✔               | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | ✔             | ✔SEFI-LW          |           |
| **ResNet**            | resnet50        | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **ResNet**            | resnet50        | CIFAR-10      | ✔               | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | ✔             | ✔SEFI-LW          |           |
| **ResNet**            | resnet56        | CIFAR-100     | Come Soon       | [PyTorch CIFAR Models](https://github.com/chenyaofo/pytorch-cifar-models) | Planned       | ✔SEFI-LW          | 72.63%    |
| **ResNet**            | resnet19light   | Fashion-MNIST | Come Soon       | [CNN-for-Fashion-MNIST](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST) | Planned       | ✔SEFI-LW          |           |
| **DenseNet**          | densenet161*    | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **DenseNet**          | densenet161     | CIFAR-10      | ✔               | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | ✔             | ✔SEFI-LW          |           |
| **SqueezeNet**        | squeezenet1_1   | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **MobileNetV3**       | v3_large        | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **MobileNetV2**       | N/A             | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **MobileNetV2**       | N/A             | CIFAR-10      | ✔               | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | Planned       | ✔SEFI-LW          |           |
| **MobileNetV2**       | v2_x1_0         | CIFAR-100     | ✔               | [PyTorch CIFAR Models](https://github.com/chenyaofo/pytorch-cifar-models) | Planned       | ✔SEFI-LW          | 73.61%    |
| **ShuffleNetV2**      | v2_x2_0         | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **ShuffleNetV2**      | v2_x2_0         | CIFAR-10      | Come Soon       | [PyTorch CIFAR Models](https://github.com/chenyaofo/pytorch-cifar-models) | N/A           | Waiting Upload    |           |
| **ShuffleNetV2**      | v2_x2_0         | CIFAR-100     | ✔               | [PyTorch CIFAR Models](https://github.com/chenyaofo/pytorch-cifar-models) | Planned       | ✔SEFI-LW          |           |
| **MNASNet**           | mnasnet1_3      | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **EfficientNetV2**    | v2_s            | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **VisionTransformer** | vit_b_32        | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **RegNet**            | y_8gf           | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **SwinTransformer**   | swin_s          | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **ConvNext**          | convnext_base   | ImageNet      | ✔               | Torchvision                                                  | ✔             | ✔Official         |           |
| **WideResNet**        | wideresnet34_10 | CIFAR-10      | Come Soon       | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) | ✔             | Waiting Upload    |           |

### What attack methods do we support？
We support 22 common attack methods, including:

| Attack Methods                               | Method Type            | Attack Approach    | Not Support Models        | Provide default parameters? |
| -------------------------------------------- | ---------------------- | ------------------ | ------------------------- | --------------------------- |
| **FGSM**                                     | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **JSMA**                                     | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **DeepFool**                                 | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **I-FGSM (****BIM****)**                     | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **C&W Attack**                               | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **Projected** **Gradient Descent** **(PGD)** | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **MI-FGSM (MIM)**                            | Transferable Black-box | Transfer, Gradient | None                      | Yes, applicable to ImageNet |
| **SI-FGSM (SIM)**                            | Transferable Black-box | Transfer, Gradient | None                      | Yes, applicable to ImageNet |
| **NI-FGSM (NIM)**                            | Transferable Black-box | Transfer, Gradient | None                      | Yes, applicable to ImageNet |
| **VMI-FGSM (VMIM)**                          | Transferable Black-box | Transfer, Gradient | None                      | Yes, applicable to ImageNet |
| **Elastic-Net Attack (EAD)**                 | White-Box              | Gradient           | None                      | Yes, applicable to ImageNet |
| **SSAH**                                     | White-Box              | Gradient           | InceptionV3、SwinT、ViT   | Yes, applicable to ImageNet |
| **One-pixel Attack (OPA)**                   | Black-Box              | Query, Score       | Not tested                | No                          |
| **Local Search Attack (LSA)**                | Black-Box              | Query, Score       | None                      | Yes, applicable to ImageNet |
| **Boundary Attack (BA)**                     | Black-Box              | Query, Decision    | None                      | Yes, applicable to ImageNet |
| **Spatial Attack (SA)**                      | Black-Box              | Query              | None                      | Yes, applicable to ImageNet |
| **Hop Skip Jump Attack (HSJA)**              | Black-Box              | Query, Decision    | None                      | Yes, applicable to ImageNet |
| **Gen Attack (GA)**                          | Black-Box              | Query, Score       | None                      | Yes, applicable to ImageNet |
| **SPSA**                                     | Black-Box              | Query, Score       | None                      | Yes, applicable to ImageNet |
| **Zeroth-Order** **Optimization** **(ZOO)**  | Black-Box              | Query, Score       | Not tested                | No                          |
| **AdvGan**                                   | Black-Box              | Query, Score       | None                      | Yes, applicable to ImageNet |
| **TREMBA**                                   | Black-Box              | Query, Score       | GoogLeNet、EfficientNetV2 | Yes, applicable to ImageNet |

We are looking for more good and classic attack methods to add to our library, if you are the author of a method, feel free to contribute your method. Some of the methods that are not fully tested may not be shown in the above table, but it may appear in the code earlier, if it fails to appear in the above list it means it may not be stable or have stable support for the time being.

### What defense methods do we support? (Experimental)
**The entire defense module is currently experimental, which means they may not be stable.**

We support 8 common defense methods, including:

| Defense Methods | Method Type          | Not Support Models | Provide default parameters? |
| --------------- | -------------------- | ------------------ | --------------------------- |
| NAT             | Adversarial Training |                    |                             |
| Mart            | Adversarial Training |                    |                             |
| Natural         | Adversarial Training |                    |                             |
| Trades          | Adversarial Training |                    |                             |
| Jpeg            | Image Processing     |                    |                             |
| Quantize        | Image Processing     |                    |                             |
| TVM             | Image Processing     |                    |                             |
| Quilting        | Image Processing     |                    |                             |

We are looking for more good and classic defense methods to add to our library, if you are the author of a method, feel free to contribute your method. Some of the methods that are not fully tested may not be shown in the above table, but it may appear in the code earlier, if it fails to appear in the above list it means it may not be stable or have stable support for the time being.

### What data do we support collecting？
We support the full collection of the following four types of metrics. Please refer to our Paper section 3.1 or the user manual for the specific meaning of the metrics.

#### Model capability measurement metrics
* Clean Example Accuracy (Clear Accuracy, CA)
* Clean example F1 score (Clear F1, CF)
* Clear Confidence (CC)

#### Attack effectiveness measurement metrics
* Misclassification Ratio (MR) /  Targeted Attack Success (TAS)
* Adversarial Example Confidence Change (ACC): Average Increase in Adversarial-class Confidence (AIAC) / Average Reduction in True-class Confidence (ARTC)
* Average Class Activation Mapping Change (ACAMC)
* Observable Transfer Rate (OTR)

#### Cost of attack measurement metrics
* Calculation Time Cost (CTC)
* Query Number Cost (QNC)
* Average Norm Distortion(AND): Average Maximum Distortion (AMD) / Average Euclidean Distortion (AED) / Average Pixel Change Ratio (APCR)
* Average Euclidean Distortion in Frequency Domain (AED-FD)
* Average Metrics Similarity (AMS): Average Deep Metrics Similarity (ADMS) / Average Low-level Metrics Similarity (ALMS)

#### Effectiveness of defense measurement metrics
* Model Capability Variance (MCV): Accuracy Variance (AV) / F1-Score Variance (FV) / Mean Confidence Variance (CV)
* Rectify/Sacrifice Ratio (RR/SR)
* Attack Capability Variance (ACV): MR Variance (MRV) / AND Variance (ANDV) / AMS Variance (AMSV)
* Average Adversarial Confidence Change (AACC):Average Reduction in Adversarial-class Confidence (ARAC) / Average Increase in True-class Confidence (AITC)

## Maintainers
[@NeoSunJz](https://github.com/NeoSunJz).

## Contributors
Our main contributors are：**孙家正（Jiazheng Sun）、Li Chen、Chenxiao Xia、Da Zhang、 Rong Huang、Zhi Qu、Wenqi Xiong**

We are particularly grateful for：**Jun Zheng 、Yu’an Tan**

## Cite
We sincerely hope that Canary can be helpful to you, and we also welcome you to cite our articles when using Canary to complete your research work:
```
@Article{electronics12173665,
  AUTHOR = {Sun, Jiazheng and Chen, Li and Xia, Chenxiao and Zhang, Da and Huang, Rong and Qiu, Zhi and Xiong, Wenqi and Zheng, Jun and Tan, Yu-An},
  TITLE = {CANARY: An Adversarial Robustness Evaluation Platform for Deep Learning Models on Image Classification},
  JOURNAL = {Electronics},
  VOLUME = {12},
  YEAR = {2023},
  NUMBER = {17},
  ARTICLE-NUMBER = {3665},
  URL = {https://www.mdpi.com/2079-9292/12/17/3665},
  ISSN = {2079-9292},
  DOI = {10.3390/electronics12173665}
}
```

## License
[Apache 2.0](LICENSE) © Beijing Institute of Technology (BIT)
