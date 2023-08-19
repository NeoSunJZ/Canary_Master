# SEFI
<img src="https://github.com/NeoSunJZ/Canary_Master/blob/main/logo.png?raw=true" width="200" alt="">

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Introduction

SEFI is a framework for evaluating the robustness of artificial intelligence. It will generate adversarial samples based on selected models using selected attack methods and use these adversarial samples to attack any model you wish. In the process it will collect data including adversarial sample quality, model result deflection and model baseline inference as a basis for evaluating the robustness of AI models and the effectiveness of attack methods, while finding the best defense solution.

It additionally provides a toolkit with multiple models, SOTA attack methods and defense methods, and allows users to integrate more on their own.

SEFI is created and maintained by researchers at Beijing Institute of Technology.

## Install

This project uses [python](https://www.python.org/), [pip](https://pypi.org/project/pip/) and [pytorch](https://pytorch.org/). Go check them out if you don't have them locally installed.

We recommend using [conda](https://github.com/conda/conda) to create a python environment rather than installing python directly.

```sh
pip install sefi
```

## Usage


## Supported

### Model

| Model Name |    Dataset    |                         Model Source                         | Best Acc | Support Grad-CAM |
| :--------: | :-----------: | :----------------------------------------------------------: | :------: | :--------------: |
|  AlexNet   | Fashion-MNIST | [CNN-for-Fashion-MNIST](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST) |  92.19%  |                  |
|  AlexNet   |   ImageNet    |                         Torchvision                          |          |        ✔         |
|    VGG     | Fashion-MNIST | [CNN-for-Fashion-MNIST](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST) |  91.35%  |                  |
|    VGG     |   CIFAR-10    |                                                              |          |        ✔         |
|    VGG     |   ImageNet    |                         Torchvision                          |          |        ✔         |
|            |               |                                                              |          |                  |



### Attack Method



### Database



## Maintainers

[@NeoSunJz](https://github.com/NeoSunJz).

## Contributing

### Contributors

This project exists thanks to all the people who contribute.

## License

[Apache 2.0](LICENSE) © Beijing Institute of Technology