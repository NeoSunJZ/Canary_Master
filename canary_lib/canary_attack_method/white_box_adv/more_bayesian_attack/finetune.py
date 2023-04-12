import copy
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader


class MoreBayesianAttackModelFinetune():
    def __init__(self, model, run_device, dataset_train, dataset_val,
                 batch_size=1024, lam=1, lr=0.05, epochs=10, default=0, swa_start=0,
                 swa_n=300, seed=0):
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.default = default
        self.swa_start = swa_start
        self.swa_n = swa_n
        self.seed = seed

        self.model = model
        self.device = run_device

        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )

    def __call__(self, *args, **kwargs):
        model = self.model
        model = nn.DataParallel(model)
        model = model.cuda()

        mean_model = copy.deepcopy(model)
        sqmean_model = copy.deepcopy(model)

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        n_ensembled = 0
        for epoch in range(self.epochs):
            for i, (img, label) in enumerate(self.train_loader):
                img, label = img.to(self.device), label.to(self.device)
                model.train()

                output_cln = model(img)
                loss_normal = F.cross_entropy(output_cln, label)
                optimizer.zero_grad()
                loss_normal.backward()
                grad_normal = self.get_grad(model)
                norm_grad_normal = self.cat_grad(grad_normal).norm()

                self.add_into_weights(model, grad_normal, gamma=+0.1 / (norm_grad_normal + 1e-20))
                loss_add = F.cross_entropy(model(img), label)
                optimizer.zero_grad()
                loss_add.backward()
                grad_add = self.get_grad(model)
                self.add_into_weights(model, grad_normal, gamma=-0.1 / (norm_grad_normal + 1e-20))

                optimizer.zero_grad()
                grad_new_dict = OrderedDict()
                for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                    grad_new_dict[name] = g_normal + (self.lam / 0.1) * (g_add - g_normal)
                self.assign_grad(model, grad_new_dict)
                optimizer.step()

                if i % 100 == 0:
                    acc = 100 * (output_cln.argmax(1) == label).sum() / len(img)
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.2f}'.format(
                        epoch, i * len(img), len(self.train_loader.dataset),
                               100. * i / len(self.train_loader), loss_normal.item(), acc))

                # SWAG
                if ((epoch + 1) > self.swa_start
                        and ((epoch - self.swa_start) * len(self.train_loader) + i) % (
                                ((self.epochs - self.swa_start) * len(self.train_loader)) // self.swa_n) == 0):
                    self.update_swag_model(model, mean_model, sqmean_model, n_ensembled)
                    n_ensembled += 1

            loss_cln_eval, acc_eval = self.eval_imgnet(args, self.val_loader, model, self.device)
            logging.info('CURRENT EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))
            print("updating BN statistics ... ")
            self.update_bn_imgnet(self.train_loader, mean_model)
            loss_cln_eval, acc_eval = self.eval_imgnet(args, self.val_loader, mean_model, self.device)
            logging.info('SWA EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))

            return mean_model, sqmean_model, epoch
            # torch.save({"state_dict": model.state_dict(),
            #             "opt_state_dict": optimizer.state_dict(),
            #             "epoch": epoch},
            #            os.path.join(args.save_dir, 'ep_{}.pt'.format(epoch)))

    class RandomResizedCrop(T.RandomResizedCrop):
        @staticmethod
        def get_params(img, scale, ratio):
            width, height = torchvision.transforms.functional.get_image_size(img)
            area = height * width

            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            w = min(w, width)
            h = min(h, height)

            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()

            return i, j, h, w

    @staticmethod
    def update_swag_model(model, mean_model, sqmean_model, n):
        for param, param_mean, param_sqmean in zip(model.parameters(), mean_model.parameters(), sqmean_model.parameters()):
            param_mean.data.mul_(n / (n + 1.)).add_(param, alpha=1. / (n + 1.))
            param_sqmean.data.mul_(n / (n + 1.)).add_(param ** 2, alpha=1. / (n + 1.))

    @staticmethod
    def update_bn_imgnet(loader, model, device=None):
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
        if not momenta:
            return
        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0
        for i, input in enumerate(loader):
            # using 10% of the training data to update batch-normalization statistics
            if i > len(loader) // 10:
                break
            if isinstance(input, (list, tuple)):
                input = input[0]
            if device is not None:
                input = input.to(device)
            with torch.no_grad():
                model(input)
        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)

    @staticmethod
    def add_into_weights(model, grad_on_weights, gamma):
        names_in_gow = grad_on_weights.keys()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in names_in_gow:
                    param.add_(gamma * grad_on_weights[name])

    @staticmethod
    def get_grad(model):
        grad_dict = OrderedDict()
        for name, param in model.named_parameters():
            grad_dict[name] = param.grad.data + 0
        return grad_dict

    @staticmethod
    def assign_grad(model, grad_dict):
        names_in_grad_dict = grad_dict.keys()
        for name, param in model.named_parameters():
            if name in names_in_grad_dict:
                if param.grad != None:
                    param.grad.data.mul_(0).add_(grad_dict[name])
                else:
                    param.grad = grad_dict[name]

    @staticmethod
    def cat_grad(grad_dict):
        dls = []
        for name, d in grad_dict.items():
            dls.append(d)
        return torch.cat([x.view(-1) for x in dls])

    @staticmethod
    def eval_imgnet(args, val_loader, model, device):
        loss_eval = 0
        # grad_norm_eval = 0
        acc_eval = 0
        for i, (img, label) in enumerate(val_loader):
            img, label = img.to(device), label.to(device)
            model.eval()
            with torch.no_grad():
                output = model(img)
            loss = F.cross_entropy(output, label)
            acc = 100 * (output.argmax(1) == label).sum() / len(img)
            loss_eval += loss.item()
            acc_eval += acc
            if i == 4:
                loss_eval /= (i + 1)
                acc_eval /= (i + 1)
                break
        return loss_eval, acc_eval
