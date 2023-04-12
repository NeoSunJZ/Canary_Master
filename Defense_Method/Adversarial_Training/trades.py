import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore
from torch.autograd import Variable
import torch.optim as optim

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.handler.model_weight_handler.weight_file_io_handler import save_weight_to_temp
from canary_sefi.task_manager import task_manager
from Defense_Method.Adversarial_Training.common import adjust_learning_rate, eval_test

sefi_component = SEFIComponent()


@sefi_component.defense_class(defense_name="trades")
@sefi_component.config_params_handler(handler_target=ComponentType.DEFENSE, name="trades",
                                      handler_type=ComponentConfigHandlerType.DEFENSE_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={

                                      })
class Trades:
    def __init__(self, lr=0.1, momentum=0.9, weight_decay=2e-4, epochs=10, run_device=None, step_size=0.003,
                 epsilon=0.031, num_steps=10, beta=6.0, log_interval=5):
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
        self.log_interval = log_interval

    @sefi_component.defense(name="trades", is_inclass=True, support_model=[])
    def defense(self, defense_model, train_dataset, val_dataset, each_epoch_finish_callback=None):
        optimizer = optim.SGD(defense_model.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            defense_model.train()
            adjust_learning_rate(self.lr, optimizer, epoch)
            # adversarial training
            for index in range(len(train_dataset)):
                data, target = train_dataset[index][0].to(self.device), train_dataset[index][1].to(self.device)
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
                msg = "[ Epoch {} ] step {}/{} -loss:{:.4f}.".format(epoch, index, len(train_dataset), loss)
                reporter.console_log(msg, Fore.GREEN, show_task=False, show_step_sequence=False)

            if epoch % self.log_interval == 0:
                model_name = defense_model.__class__.__name__ + "(CIFAR-10)"
                file_name = "AT_" + "trades" + '_' + model_name + "_" + "CIFAR-10" + "_" + task_manager.task_token + "_" + str(
                    epoch) + ".pt"
                save_weight_to_temp(file_path=model_name + '/trades/', file_name=file_name, weight=defense_model.state_dict())

            # val预测
            eval_test(val_dataset, defense_model, epoch, self.device)
            each_epoch_finish_callback(epoch)

        return defense_model.state_dict()

    def trades_loss(self, model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
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

    # def eval_test(self, val_dataset, defense_model, epoch):
    #     acc = 0
    #     val_loss = 0
    #     for index in range(len(val_dataset)):
    #         defense_model.eval()
    #         data, target = val_dataset[index][0].to(self.device), val_dataset[index][1].to(self.device)
    #         logit = defense_model(data)
    #         val_loss += F.cross_entropy(logit, target, size_average=False).item()
    #         output = F.softmax(logit, dim=1)
    #         pred_label = torch.argmax(output, dim=1)
    #         acc += torch.sum(torch.eq(pred_label, target))
    #
    #     msg = "[ Val ] epoch {} -val_loss:{:.4f} -acc:{:.4f}.".format(epoch, val_loss / val_dataset.dataset_size,
    #                                                                   acc / val_dataset.dataset_size)
    #     reporter.console_log(msg, Fore.GREEN, show_task=False, show_step_sequence=False)
    #
    # def adjust_learning_rate(self, optimizer, epoch):
    #     """decrease the learning rate"""
    #     lr = self.lr
    #     if epoch >= 75:
    #         lr = self.lr * 0.1
    #     if epoch >= 90:
    #         lr = self.lr * 0.01
    #     if epoch >= 100:
    #         lr = self.lr * 0.001
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
