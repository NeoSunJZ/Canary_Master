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
class Trades():
    def __int__(self):
        pass

    @sefi_component.defense(name="trades", is_inclass=True, support_model=[])
    def defense(self, defense_model, img_preprocessor, img_reverse_processor, img_proc_args_dict, ori_dataset):
        self.model.train()
        base_path = "../../Model_Save/"
        for epoch in range(1, self.args.epochs + 1):
            # adjust learning rate for SGD
            self.adjust_learning_rate(epoch)

            # adversarial training
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # calculate robust loss
                loss = self.trades_loss(model=self.model,
                                        x_natural=data,
                                        y=target,
                                        optimizer=self.optimizer,
                                        step_size=self.args.step_size,
                                        epsilon=self.args.epsilon,
                                        perturb_steps=self.args.num_steps,
                                        beta=self.args.beta)
                loss.backward()
                self.optimizer.step()

                # print progress
                if batch_idx % self.args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))
            # Save model
            torch.save(self.model.state_dict(),
                       os.path.join(base_path, self.args.dataset + self.args.model +
                                    '_baseline_epoch_' + str(epoch) + '.pt'))
            # Let us not waste space and delete the previous model
            prev_path = os.path.join(base_path, self.args.dataset + self.args.model +
                                     '_baseline_epoch_' + str(epoch - 1) + '.pt')
            if os.path.exists(prev_path):
                os.remove(prev_path)

    def trades_loss(self,
                    model,
                    x_natural,
                    y,
                    optimizer,
                    step_size=0.003,
                    epsilon=0.031,
                    perturb_steps=10,
                    beta=1.0,
                    distance='l_inf'):
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

    def adjust_learning_rate(self, epoch):
        """decrease the learning rate"""
        lr = self.args.lr
        if epoch >= 75:
            lr = self.args.lr * 0.1
        if epoch >= 90:
            lr = self.args.lr * 0.01
        if epoch >= 100:
            lr = self.args.lr * 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
