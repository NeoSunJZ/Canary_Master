import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.defense_class(defense_name="Madry")
@sefi_component.config_params_handler(handler_target=ComponentType.DEFENSE, name="Madry",
                                      handler_type=ComponentConfigHandlerType.DEFENSE_CONFIG_PARAMS, use_default_handler=True,
                                      params={

                                      })
class Trades():
    def __init__(self, lr=0.1, momentum=0.9, weight_decay=2e-4, epochs=10, run_device=None, step_size=0.003,
                 epsilon=0.031, num_steps=10, beta=6.0):
        # defense_args_dict传到这里初始化
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = run_device
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.beta = beta

    @sefi_component.defense(name="Madry", is_inclass=True, support_model=[])
    def defense(self, defense_model, dataset, each_epoch_finish_callback=None):
        defense_model.train()
        optimizer = optim.SGD(defense_model.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        # for epoch in range(1, self.epochs + 1):
        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            self.adjust_learning_rate(optimizer, epoch)

            # adversarial training
            for index in range(len(dataset)):
                # print(len(dataset))
                # data, target = imgs[index].to(self.device), labels[index].to(self.device)
                data, target = dataset[index][0].to(self.device), dataset[index][1].to(self.device)
                optimizer.zero_grad()
                # calculate robust loss
                loss = self.madry_loss(model=defense_model,
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
                # if index % self.log_interval == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, index * len(data), len(imgs),
                #                100. * index / len(imgs), loss.item()))

            # print(epoch, each_epoch_finish_callback)
            each_epoch_finish_callback(epoch)

        return defense_model.state_dict()

    def madry_loss(self, model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                   distance='l_inf'):

        criterion = nn.CrossEntropyLoss()
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    ce_loss = criterion(model(x_adv), y)
                grad = torch.autograd.grad(ce_loss, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_adv), y)
        return loss

    def adjust_learning_rate(self, optimizer, epoch):
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
