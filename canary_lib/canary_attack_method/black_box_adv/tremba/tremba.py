import os.path

import torch
import numpy as np
from torch import nn

from canary_lib.canary_attack_method.black_box_adv.tremba.fcn import Imagenet_Encoder, Imagenet_Decoder
from canary_lib.canary_attack_method.black_box_adv.tremba.train_generator import train_generator
from canary_lib.canary_attack_method.black_box_adv.tremba.utils import Function
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentConfigHandlerType, ComponentType
from canary_sefi.core.component.default_component.model_getter import get_model
from canary_sefi.task_manager import task_manager

sefi_component = SEFIComponent()

@sefi_component.attacker_class(attack_name="TREMBA", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="TREMBA",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS, use_default_handler=True,
                                      params={})
class TREMBA():
    def __init__(self, model, run_device, nlabels=1000, epsilon=0.03, attack_type='UNTARGETED'):
        self.run_device = run_device
        self.model = model

        self.is_target = attack_type == 'TARGETED'
        self.nlabels = nlabels
        self.epsilon = epsilon
        self.margin = 5.0
        self.sample_size = 20
        self.num_iters = 200
        self.sigma = 1.0
        self.lr = 5.0
        self.lr_min = 0.1
        self.lr_decay = 2.0
        self.momentum = 0.0
        self.plateau_length = 20
        self.plateau_overhead = 0.3
        self.batch_size = 32

        # 训练
        self.train_epochs = 200
        self.train_learning_rate_G = 0.01
        self.train_momentum_G = 0.9
        self.train_schedule_G = 10
        self.train_gamma_G = 0.75
        self.train_margin = 200.0
        self.train_target_class = None

        self.weight_save_path = task_manager.base_temp_path + "weight/attack/tremba/"
        self.weight_save_name = None
        self.weight = None

    @sefi_component.attack_init(name="TREMBA")
    def train(self, dataset_info, dataset_loader, batch_size, model_name):
        self.weight_save_name = "Imagenet_{}_{}.pt" \
            .format(model_name, "untarget" if not self.is_target else "target_{}".format(self.train_target_class))
        if os.path.exists(self.weight_save_path + self.weight_save_name):
            self.weight = torch.load(self.weight_save_path + self.weight_save_name, map_location=self.run_device)
        if self.weight is None:
            net = [get_model(model_name, {}, self.run_device)]
            train_generator(self.run_device, net, [model_name], dataset_loader(dataset_info),
                            batch_size, self.train_epochs,
                            self.train_learning_rate_G, self.train_momentum_G,
                            self.train_schedule_G, self.train_gamma_G,
                            self.train_margin, self.is_target, self.train_target_class, self.epsilon,
                            weight_save_path=self.weight_save_path, weight_save_name=self.weight_save_name)
            self.weight = torch.load(self.weight_save_path + self.weight_save_name, map_location=self.run_device)
        else:
            print("Already has weight!")

    @sefi_component.attack(name="TREMBA", is_inclass=True, support_model=[], attack_type="BLACK_BOX")
    def attack(self, imgs, ori_labels):
        encoder_weight = {}
        decoder_weight = {}
        for key, val in self.weight.items():
            if key.startswith('0.'):
                encoder_weight[key[2:]] = val
            elif key.startswith('1.'):
                decoder_weight[key[2:]] = val

        encoder = Imagenet_Encoder()
        decoder = Imagenet_Decoder()
        encoder.load_state_dict(encoder_weight)
        decoder.load_state_dict(decoder_weight)

        self.model.to(self.run_device).eval()
        encoder.to(self.run_device).eval()
        decoder.to(self.run_device).eval()

        F = Function(self.model, self.batch_size, self.margin, self.nlabels, self.is_target)

        if self.is_target:
            labels = np.repeat(self.train_target_class, imgs.shape[0])
        else:
            labels = ori_labels

        adv_images = None
        for i in range(imgs.shape[0]):
            with torch.no_grad():
                success, adv = self.EmbedBA(F, encoder, decoder, imgs[i], labels[i])
                adv_img = torch.unsqueeze(adv, dim=0)
                if adv_images is None:
                    adv_images = adv_img
                else:
                    adv_images = torch.cat((adv_images, adv_img), dim=0)
        return adv_images


    def EmbedBA(self, function, encoder, decoder, image, label, latent=None):
        if latent is None:
            latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
        momentum = torch.zeros_like(latent)
        dimension = len(latent)
        noise = torch.empty((dimension, self.sample_size), device=self.run_device)
        origin_image = image.clone()
        last_loss = []
        lr = self.lr

        for iter in range(self.num_iters + 1):
            perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0) * self.epsilon, -self.epsilon, self.epsilon)
            logit, loss = function(torch.clamp(image + perturbation, 0, 1), label)
            if self.is_target:
                success = torch.argmax(logit, dim=1) == label
            else:
                success = torch.argmax(logit, dim=1) != label
            last_loss.append(loss.item())

            if function.current_counts > 50000:
                break

            if bool(success.item()):
                return True, torch.clamp(image + perturbation, 0, 1)

            nn.init.normal_(noise)
            noise[:, self.sample_size // 2:] = -noise[:, :self.sample_size // 2]
            latents = latent.repeat(self.sample_size, 1) + noise.transpose(0, 1) * self.sigma
            perturbations = torch.clamp(decoder(latents) * self.epsilon, -self.epsilon, self.epsilon)
            _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

            grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

            print("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()),
                                                                   float(torch.norm(perturbation))))

            momentum = self.momentum * momentum + (1 - self.momentum) * grad

            latent = latent - lr * momentum

            last_loss = last_loss[-self.plateau_length:]
            if (last_loss[-1] > last_loss[0] + self.plateau_overhead or last_loss[-1] > last_loss[0] and last_loss[-1] < 0.6) \
                    and len(last_loss) == self.plateau_length:
                if lr > self.lr_min:
                    lr = max(lr / self.lr_decay, self.lr_min)
                last_loss = []
        return False, origin_image

