import os
import numpy as np
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
@sefi_component.config_params_handler(handler_target=ComponentType.DEFENSE, name="trades",
                                      args_type=ComponentConfigHandlerType.DEFENSE_PARAMS, use_default_handler=True,
                                      params={

                                      })

class Trades():
    def __init__(self, lr=0.1, momentum=0.9, weight_decay=2e-4, epochs=10, device="cuda", step_size = 0.007,
                epsilon = 0.031, num_steps = 10, beta = 6.0, log_interval = 5):
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
    def defense(self, defense_model, imgs, labels):
        defense_model.train()
        optimizer = optim.SGD(defense_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # for epoch in range(1, self.epochs + 1):
        for epoch in range(1):
            # adjust learning rate for SGD
            self.adjust_learning_rate(optimizer,epoch)

            # adversarial training
            for index in range(len(imgs)):
                #print(np.array(data).shape)
                #data = img_preprocessor(data, img_proc_args_dict)
                data, target = imgs[index].to(self.device), labels[index].to(self.device)

                optimizer.zero_grad()
                print("x_natural_{}:{}".format(index,data.tolist()))
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

                # if index==10:
                #     with open("x_natural_10.txt",'w') as output:
                #         output.write(str(data.tolist()))
                #         output.close()

                # print progress
            # print("epoch:{},loss:{:.6f}".format(epoch,loss.item()))
                if index % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, index * len(data), len(imgs),
                               100. * index / len(imgs), loss.item()))

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
        print("输出:",logits)
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
