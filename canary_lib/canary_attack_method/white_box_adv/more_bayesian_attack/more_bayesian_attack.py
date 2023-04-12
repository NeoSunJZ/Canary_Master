import copy
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from canary_lib.canary_attack_method.white_box_adv.more_bayesian_attack.finetune import MoreBayesianAttackModelFinetune

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType
from canary_sefi.entity.dataset_info_entity import DatasetInfo

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="MBA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="MBA",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS, use_default_handler=True,
                                      params={})
class MoreBayesianAttack():
    def __init__(self, run_device, model, attack_type="UNTARGETED",
                 epsilon=8/255, step_size=1/255, niters=50, batch_size=500, force="store_true", seed=0,
                 constraint="linf", n_models=20, beta=0, scale=1.5,
                 dataset_train_name=None, dataset_val_name=None):
        self.epsilon = epsilon
        self.step_size = step_size
        self.niters = niters
        self.batch_size = batch_size
        self.force = force
        self.seed = seed
        self.constraint = constraint
        self.n_models = n_models
        self.beta = beta
        self.scale = scale
        self.dataset_train_name = dataset_train_name
        self.dataset_val_name = dataset_val_name

        self.device = run_device
        self.normalize = model[0]
        self.model = model[1]

        # self.weight_save_path = task_manager.base_temp_path + "weight/attack/mba/"
        self.weight_save_path = os.path.dirname(__file__) + "/"
        self.mean_model = None
        self.sqmean_model = None

    def build_model(self, model, state_dict):
        if "module" in list(state_dict.keys())[0]:
            model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.module
        else:
            model.load_state_dict(state_dict)
        model.eval()
        model = nn.Sequential(self.normalize, model)
        model.to(self.device)
        return model

    @sefi_component.attack_init(name="MBA")
    def train(self, dataset_info, dataset_loader, batch_size, model_name):
        weight_save_name = '{}_morebayesian_attack.pt'.format(model_name)
        def load_model():
            state_dict = torch.load(self.weight_save_path + weight_save_name)
            self.mean_model = self.build_model(self.model, state_dict["mean_state_dict"])
            self.sqmean_model = self.build_model(self.model, state_dict["sqmean_state_dict"])
            self.mean_model = nn.DataParallel(self.mean_model)
        if os.path.exists(self.weight_save_path + weight_save_name):
            load_model()
            return
        else:
            dataset_info_train = DatasetInfo(dataset_name=self.dataset_train_name)
            dataset_info_val = DatasetInfo(dataset_name=self.dataset_val_name)
            mean_model, sqmean_model, epoch = MoreBayesianAttackModelFinetune(
                self.model, run_device=self.device, dataset_train=dataset_loader(dataset_info_train),
                dataset_val=dataset_loader(dataset_info_val))

            torch.save({"mean_state_dict": mean_model.state_dict(),
                        "sqmean_state_dict": sqmean_model.state_dict(),
                        "epoch": epoch}, os.path.join(self.weight_save_path, weight_save_name))
            load_model()
            return

    def get_model_list(self, mean_model, sqmean_model):
        model_list = []
        for model_ind in range(self.n_models):
            model_list.append(copy.deepcopy(mean_model))
            var_avg = 0
            c = 0
            noise_dict = OrderedDict()
            for (name, param_mean), param_sqmean, param_cur in zip(mean_model.named_parameters(),
                                                                   sqmean_model.parameters(),
                                                                   model_list[-1].parameters()):
                var = torch.clamp(param_sqmean.data - param_mean.data ** 2, 1e-30)
                var = var + self.beta
                noise_dict[name] = var.sqrt() * torch.randn_like(param_mean, requires_grad=False)
                c += param_mean.numel()

            for (name, param_cur), (_, noise) in zip(model_list[-1].named_parameters(), noise_dict.items()):
                param_cur.data.add_(noise, alpha=self.scale)
        return model_list

    @sefi_component.attack(name="MBA", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_labels, tlabels=None):
        ori_imgs, labels = imgs.to(self.device), torch.from_numpy(np.array(ori_labels)).to(self.device)

        batch_size = len(ori_imgs)
        att_imgs = ori_imgs.clone()
        for i in range(self.niters):
            model_list = self.get_model_list(self.mean_model, self.sqmean_model)
            input_grad, loss = self.get_input_grad(att_imgs, labels, model_list)
            att_imgs = self.update_and_clip(ori_imgs, att_imgs, input_grad, self.epsilon, self.step_size, self.constraint)
            print('iter {}, loss {:.4f}'.format(i, loss), end='\n')
        return att_imgs

    @staticmethod
    def get_input_grad(x, y, model_list):
        model = random.choice(model_list)
        x.requires_grad_(True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        ce_grad = torch.autograd.grad(loss, [x, ])[0]
        final_grad = ce_grad.data
        loss_sum = loss.item()
        return final_grad, loss_sum

    @staticmethod
    def update_and_clip(ori_img, att_img, grad, epsilon, step_size, norm):
        if norm == "linf":
            att_img = att_img.data + step_size * torch.sign(grad)
            att_img = torch.where(att_img > ori_img + epsilon, ori_img + epsilon, att_img)
            att_img = torch.where(att_img < ori_img - epsilon, ori_img - epsilon, att_img)
            att_img = torch.clamp(att_img, min=0, max=1)
        elif norm == "l2":
            grad = grad / (grad.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-12)
            att_img = att_img.data + step_size * grad
            l2_perturb = att_img - ori_img
            l2_perturb = l2_perturb.renorm(p=2, dim=0, maxnorm=epsilon)
            att_img = ori_img + l2_perturb
            att_img = torch.clamp(att_img, min=0, max=1)
        return att_img
