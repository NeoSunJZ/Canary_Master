import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore
from torch.autograd import Variable
import torch.optim as optim

from canary_lib.canary_defense_method.adversarial_training.common import adjust_learning_rate, eval_test
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.handler.model_weight_handler.weight_file_io_handler import save_weight_to_temp


sefi_component = SEFIComponent()


@sefi_component.defense_class(defense_name="MART")
@sefi_component.config_params_handler(handler_target=ComponentType.DEFENSE, name="MART",
                                      handler_type=ComponentConfigHandlerType.DEFENSE_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={

                                      })
class Mart:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=3.5e-3, epochs=10, run_device=None, step_size=0.007,
                 epsilon=0.031, num_steps=10, beta=5.0, log_interval=5):
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

    @sefi_component.defense(name="MART", is_inclass=True, support_model=[])
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
                loss = self.mart_loss(model=defense_model,
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
                reporter.console_log(msg, Fore.GREEN, save_db=False, show_task=False, show_step_sequence=False)

            if epoch % self.log_interval == 0:
                model_name = defense_model.__class__.__name__ + "(CIFAR-10)"
                file_name = "AT_" + "MART" + '_' + model_name + "_" + str(epoch) + ".pt"
                save_weight_to_temp(model_name=model_name, defense_name="MART", epoch_cursor=str(epoch),file_path=model_name + '/MART/',
                                    file_name=file_name, weight=defense_model.state_dict())

            # val预测
            eval_test(val_dataset, defense_model, epoch, self.device)
            each_epoch_finish_callback(epoch)

        return defense_model.state_dict()

    def mart_loss(self, model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                  distance='l_inf'):
        kl = nn.KLDivLoss(reduction='none')
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_ce = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
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
        logits_adv = model(x_adv)
        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

        loss = loss_adv + float(beta) * loss_robust
        return loss
