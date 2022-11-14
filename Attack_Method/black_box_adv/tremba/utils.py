import torch.nn as nn
import torch
import numpy as np


class MarginLoss(nn.Module):

    def __init__(self, margin=1.0, target=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.target = target

    def forward(self, logits, label):

        if not self.target:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return margin.mean()


class MarginLoss_Single(nn.Module):

    def __init__(self, margin=1.0, target=False):
        super(MarginLoss_Single, self).__init__()
        self.margin = margin
        self.target = target

    def forward(self, logits, label):

        if not self.target:
            if label == 0:
                logits_cat = logits[:,(label+1):]
            elif label == logits.size()[1] - 1:
                logits_cat = logits[:, :label]
            else:
                logits_cat = torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1)
            diff = logits[:,label] - torch.max(logits_cat, dim=1)[0]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return margin.mean()

class Function(nn.Module):

    def __init__(self, model, batch_size=256, margin=0, nlabels=10, target=False):
        super(Function, self).__init__()
        self.model = model
        self.margin = margin
        self.target = target
        self.batch_size = batch_size
        self.current_counts = 0
        self.counts = []
        self.nlabels = nlabels

    def _loss(self, logits, label):
        if not self.target:
            if label == 0:
                logits_cat = logits[:,(label+1):]
            elif label == logits.size()[1] - 1:
                logits_cat = logits[:, :label]
            else:
                logits_cat = torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1)
            diff = logits[:,label] - torch.max(logits_cat, dim=1)[0]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return margin

    def forward(self, images, label):
        if len(images.size())==3:
            images = images.unsqueeze(0)
        n = len(images)
        device = images.device
        k = 0
        loss = torch.zeros(n, dtype=torch.float32, device=device)
        logits = torch.zeros((n, self.nlabels), dtype=torch.float32, device=device)

        while k < n:
            start = k
            end = min(k + self.batch_size, n)
            logits[start:end] = self.model(images[start:end])
            loss[start:end] = self._loss(logits, label)
            k = end
        self.current_counts += n

        return logits, loss

    def new_counter(self):
        self.counts.append(self.current_counts)
        self.current_counts = 0

    def get_average(self, iter=50000):
        counts = np.array(self.counts)
        return np.mean(counts[counts<iter])