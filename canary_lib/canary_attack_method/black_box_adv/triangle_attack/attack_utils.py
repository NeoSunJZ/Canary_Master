import torch
import random
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from foolbox import PyTorchModel
import torchvision.models as models
from datetime import datetime
import pandas as pd

def get_model(args,device):
    model_name = args.model_name
    if model_name == 'resnet-18':
        model = models.resnet18(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'inception-v3':
        model = models.inception_v3(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'vgg-16':
        model = models.vgg16(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'resnet-101':
        model = models.resnet101(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'densenet-121':
        model = models.densenet121(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel


def get_label(logit):
    _, predict = torch.max(logit, 1)
    return predict



def save_results(args,my_intermediates, n):
    path = args.output_folder
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path):
        os.mkdir(path)
    numpy_results = np.full((n * 3, args.max_queries), np.nan)
    for i, my_intermediate in enumerate(my_intermediates):
        length = len(my_intermediate)
        for j in range(length):
            numpy_results[3 * i][j] = my_intermediate[j][0]
            numpy_results[3 * i + 1][j] = my_intermediate[j][1]
            numpy_results[3 * i + 2][j] = my_intermediate[j][2]
    pandas_results = pd.DataFrame(numpy_results)
    pandas_results.to_csv(os.path.join(path,'results.csv'))
    print('save results to:{}'.format(os.path.join(path,'results.csv')))


def read_imagenet_data_specify(args, device):
    images = []
    labels = []
    info = pd.read_csv(args.csv)
    selected_image_paths = []
    for d_i in range(len(info)):
        image_path = info.iloc[d_i]['ImageName']
        image = Image.open(os.path.join(args.dataset_path,image_path))
        image = image.convert('RGB')
        image = image.resize((args.side_length, args.side_length))
        image = np.asarray(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        groundtruth = info.iloc[d_i]['Label']
        images.append(image)
        labels.append(groundtruth)
        selected_image_paths.append(image_path)
    images = np.stack(images)
    labels = np.array(labels)
    images = images / 255
    images = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device).long()
    return images, labels, selected_image_paths

