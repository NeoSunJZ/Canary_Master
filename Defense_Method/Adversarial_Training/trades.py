import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from CANARY_SEFI.handler.tools.foolbox_adapter import FoolboxAdapter

sefi_component = SEFIComponent()


@sefi_component.defense_class(defense_name="trades")
@sefi_component.config_params_handler(handler_target=ComponentType.DEFENCE, name="trades",
                                      args_type=ComponentConfigHandlerType.DEFENCE_PARAMS, use_default_handler=True,
                                      params={

                                      })

class Trades():
    def __int__(self, lr=0.1, momentum=0.9, weight_decay=2e-4, epochs=10, device="cuda", step_size = 0.007,
                epsilon = 0.031, num_steps = 10, beta = 6.0, log_interval = 100):
        #defense_args_dict传到这里初始化
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.beta = beta
        self.log_interval = log_interval

    @sefi_component.defense(name="trades", is_inclass=True, support_model=[])
    def defense(self, defense_model, img_preprocessor, img_reverse_processor, img_proc_args_dict, ori_dataset):
        defense_model.train()
        optimizer = optim.SGD(defense_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        #ori_dataset[0] is img,[1] is label
        datasets = img_preprocessor(ori_dataset,img_proc_args_dict)
        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            self.adjust_learning_rate(epoch)

            # adversarial training
            for batch_idx, (data, target) in enumerate(datasets):

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                # calculate robust loss
                loss = self.trades_loss(model=defense_model,
                                        x_natural=data,
                                        y=target,
                                        optimizer=optimizer,
                                        step_size=self.step_size,
                                        epsilon=self.epsilon,
                                        perturb_steps=self.num_steps,
                                        beta=self.beta)
                loss.backward()
                optimizer.step()

                # print progress
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(datasets),
                               100. * batch_idx / len(datasets), loss.item()))

        return defense_model.state_dict()

    def trades_loss(self, model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(model(x_natural), dim=1))
        loss = loss_natural + beta * loss_robust
        return loss

    def adjust_learning_rate(self,optimizer, epoch):
        """decrease the learning rate"""
        lr = self.lr
        if epoch >= 75:
            lr = self.lr * 0.1
        if epoch >= 90:
            lr = self.lr * 0.01
        if epoch >= 100:
            lr = self.lr * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
