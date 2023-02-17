config = {
    "adv_example_generate_batch_config": {
        "AdvGan": {  # 3090:24G
            "Alexnet(ImageNet)": 150,  # ~8G 3090:65%
            "VGG(ImageNet)": 50,  # ~19G 3090:95%
            "GoogLeNet(ImageNet)": 80,  # ~10G 3090:85%
            "InceptionV3(ImageNet)": 50,  # ~7G 3090:90%
            "ResNet(ImageNet)": 50,  # ~11G 3090:90%
            "DenseNet(ImageNet)": 35,  # ~19G 3090:90%
            "SqueezeNet(ImageNet)": 150,  # ~12G 3090:80%
            "MobileNetV3(ImageNet)": 100,  # ~11G 3090:80%
            "ShuffleNetV2(ImageNet)": 100,  # ~10G 3090:80%
            "MNASNet(ImageNet)": 50,  # ~8G 3090:85%
            "EfficientNetV2(ImageNet)": 20,  # ~20G 3090:95%
            "ViT(ImageNet)": 50,  # ~6G 3090:90%
            "RegNet(ImageNet)": 40,  # ~18G 3090:100%
            "SwinTransformer(ImageNet)": 50,  # ~19G 3090:95%
            "ConvNext(ImageNet)": 35,  # ~18G 3090:100%
        },
    }
}