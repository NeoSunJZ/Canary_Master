import sys

from tqdm import tqdm
import json
import torch
from torchvision import transforms
import random
import numpy as np

import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    model.load_state_dict(model_spec['sd'])
    # if args is not None:
    #     model_args = copy.deepcopy(model_spec['args'])
    #     model_args.update(args)
    # else:
    #     model_args = model_spec['args']
    # model = models[model_spec['name']](**model_args)
    # if load_sd:
    #     model.load_state_dict(model_spec['sd'])
    return model

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class INR(object):
    def __init__(self, device, pretrain_inr_path, height=299, width=299):
        self.device = device
        # self.inr_model = inr_models.make(torch.load(pretrain_inr_path)['model'], load_sd=True).to(self.device)
        self.inr_model = []
        for idx in range(len(pretrain_inr_path)):
            self.inr_model.append(
                make(torch.load(pretrain_inr_path[idx])['model'], load_sd=True).to(self.device))

        self.height = height
        self.width = width

        self.coord = make_coord((self.height, self.width)).to(self.device)
        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.height
        self.cell[:, 1] *= 2 / self.width

    def batched_predict(self, inp, coord, cell, bsize):
        with torch.no_grad():
            # self.inr_model.gen_feat(inp)
            for idx in range(len(self.inr_model)):
                self.inr_model[idx].gen_feat(inp)

            n = coord.shape[1]
            ql = 0
            preds = []

            while ql < n:
                qr = min(ql + bsize, n)
                idx = random.randint(0, len(self.inr_model) - 1)
                pred = self.inr_model[idx].query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                # pred = self.inr_model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr

            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self, x):
        lst_img = []
        for img in x:
            img_tensor = img.unsqueeze(0)
            inr_output = \
            self.batched_predict(((img_tensor - 0.5) / 0.5), self.coord.unsqueeze(0), self.cell.unsqueeze(0),
                                 bsize=90000)[0]
            inr_output = (inr_output * 0.5 + 0.5).clamp(0, 1).view(self.height, self.width, 3).permute(2, 0, 1)
            lst_img.append(inr_output)

        # return x.new_tensor(torch.stack(lst_img))
        return torch.stack(lst_img)


