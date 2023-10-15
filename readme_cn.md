# Canary SEFI
<img src="https://github.com/NeoSunJZ/Canary_Master/blob/main/logo.png?raw=true" width="200" alt="">

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

[English](https://github.com/NeoSunJZ/Canary_Master/blob/main/readme.md)

## 简介

SEFI是一个评估基于深度学习的图像识别模型稳健性的框架。

它使用选定的攻击方法基于选定的模型生成对抗性样本，并使用这些对抗性样本攻击您想要的任何模型。在此过程中，它将收集包括多个评估指标的数据，以此评估人工智能模型稳健性和攻击方法有效性，同时尝试找到最佳防御解决方案。

它还提供了一个包含多个模型、SOTA攻击方法和防御方法的工具包，并允许用户自己进行更多集成。

SEFI由北京理工大学（BIT）网络空间安全学院（Cyberspace Science & Technology）的研究人员创建和维护。

<img src="https://github.com/NeoSunJZ/Canary_Master/blob/main/framework_structure.png?raw=true" width="800" alt="">

## 功能

### 我们支持了哪些模型？

我们已经支持了4种数据集上的15种模型，其中ImageNet数据集的15种模型都是可用的。我们希望大家参与完善我们的模型库，分享自己的模型结构与权重信息以帮助更多的人

我们建立了一个公开的模型权重仓库，位于：https://github.com/NeoSunJZ/Canary_SEFI_Lib_Weight

| 模型                  | 子结构          | 数据集        | 完全可用  | 来源                  | 是否支持Gard-CAM | 权重分发       | Top-1 Acc |
| --------------------- | --------------- | ------------- | --------- | --------------------- | ---------------- | -------------- | --------- |
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

### 我们支持了哪些攻击方法？

我们支持了22种常见的攻击方法，包括：

| 攻击方法                                     | 方法类型               | 攻击途径           | 不支持的模型              | 默认参数适配描述            |
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

我们正在进一步寻找更多优秀和经典的攻击方法以加入我们的库，如果你是方法的作者，欢迎贡献你的方法。一些未经完全测试的方法可能没有显示在上表中，但它可能会提前出现在代码中，如果它未能出现在上述列表中说明它可能并不稳定也暂时没有得到稳定的支持。

### 我们支持了哪些防御方法？(实验性的)

整个防御模块目前都是实验性的，这意味着它们可能并不稳定。

我们支持了8种常见的防御方法，包括：

| 防御方法 | 方法类型             | 不支持的模型 | 默认参数适配描述 |
| -------- | -------------------- | ------------ | ---------------- |
| NAT      | Adversarial Training |              |                  |
| Mart     | Adversarial Training |              |                  |
| Natural  | Adversarial Training |              |                  |
| Trades   | Adversarial Training |              |                  |
| Jpeg     | Image Processing     |              |                  |
| Quantize | Image Processing     |              |                  |
| TVM      | Image Processing     |              |                  |
| Quilting | Image Processing     |              |                  |

我们正在进一步寻找更多优秀和经典的防御方法以加入我们的库，如果你是方法的作者，欢迎贡献你的方法。一些未经完全测试的方法可能没有显示在上表中，但它可能会提前出现在代码中，如果它未能出现在上述列表中说明它可能并不稳定也暂时没有得到稳定的支持。

### 我们支持采集哪些数据？

我们支持以下四类指标的完全采集。针对指标的具体意义请参考我们的论文3.1章节或用户手册。

#### 模型能力评估指标 (Model capability measurement metrics)

- 干净样本正确率 (Clear Accuracy, CA)

- 干净样本F1分数 (Clear F1, CF)

- 干净样本平均正确分类置信度 (Clear Confidence, CC)

#### 攻击效果评估指标 (Attack effectiveness measurement metrics)

- 误分类比例 (Misclassification Ratio, MR) /  攻击准确性 (Targeted Attack Success, TAS)

- 对抗样本置信偏移 (Average Confidence Change, ACC): 抗类平均置信增高(Average Increase in Adversarial-class Confidence, AIAC)、真实类平均置度降低(Average Reduction in True-class Confidence, ARTC)

- 对抗样本模型关注区域偏移 (Average Class Activation Mapping Change, ACAMC)

- 对抗样本可观测转移率 (Observable Transfer Rate, OTR)

#### 攻击代价评估指标 (Cost of attack measurement metrics)

- 攻击耗时代价 (Calculation Time Cost, CTC)

- 攻击查询量代价 (Query Number Cost, QNC)

- 平均范数距离 (Average Norm Distortion, AND): 最大像素距离(Average Maximum Distortion, AMD)、平均欧式距离(Average Euclidean Distortion, AED)、像素变化比例(Average Pixel Change Ratio, APCR)

- 平均频域欧式距离 (Average Euclidean Distortion in Frequency Domain, AED-FD)

- 平均特征相似性 (Average Metrics Similarity, AMS): 平均深度特征相似性(Average Deep Metrics Similarity, ADMS)、平均低层特征相似性(Average Low-level MetricsSimilarity, ALMS)

#### 防御效果评估指标 (Effectiveness of defense measurement metrics)

- 模型能力损失 (Model Capability Variance, MCV): 准确率差异（(Accuracy Variance, AccV)、F1-Score差异(F1-Score Variance, F1V)、平均置信差异(Confidence Variance, ConfV)

- 矫正/牺牲比率 (Rectify/Sacrifice Ratio, RR/SR)

- 攻击能力弱化(Attack Capability Variance, ACV): MR差异(MR Variance, MRV)、AND差异(AND Variance, ANDV)、AMS差异(AMS Variance, AMSV)

- 对抗置信偏移(Average Adversarial Confidence Change, AACC): 对抗类平均置信降低(Average Reduction in Adversarial-class Confidence, ARAC)、真实类平均置度升高(Average Increase in True-class Confidence, AITC)


## 快速使用

请参考工程中的 Quick Start Example

我们正在准备用户手册，并将于最近发布。

## 维护者

[@NeoSunJz](https://github.com/NeoSunJz).

## 贡献者

我们的主要贡献者是：**孙家正（Jiazheng Sun）、Li Chen、Chenxiao Xia、Da Zhang、 Rong Huang、Zhi Qu、Wenqi Xiong**

我们格外致谢：**Jun Zheng 、Yu’an Tan**

## 许可证

[Apache 2.0](LICENSE) © 北京理工大学
