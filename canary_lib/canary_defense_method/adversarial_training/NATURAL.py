import torch.nn.functional as F
from colorama import Fore
import torch.optim as optim

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.handler.model_weight_handler.weight_file_io_handler import save_weight_to_temp
from canary_lib.canary_defense_method.adversarial_training.common import adjust_learning_rate, eval_test

sefi_component = SEFIComponent()


@sefi_component.defense_class(defense_name="NATURAL")
@sefi_component.config_params_handler(handler_target=ComponentType.DEFENSE, name="NATURAL",
                                      handler_type=ComponentConfigHandlerType.DEFENSE_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={

                                      })
class Natural:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=3.5e-3, epochs=10, run_device=None, log_interval=5):
        # defense_args_dict传到这里初始化
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = run_device
        self.log_interval = log_interval

    @sefi_component.defense(name="NATURAL", is_inclass=True, support_model=[])
    def defense(self, defense_model, train_dataset, val_dataset, each_epoch_finish_callback=None):
        optimizer = optim.SGD(defense_model.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            defense_model.train()
            adjust_learning_rate(self.lr, optimizer, epoch)
            for index in range(len(train_dataset)):
                dataset = train_dataset[index]
                data, target = dataset[0].to(self.device), dataset[1].to(self.device)
                optimizer.zero_grad()
                loss = self.loss(model=defense_model,
                                 x_natural=data,
                                 y=target,
                                 optimizer=optimizer
                                 )
                loss.backward()
                optimizer.step()
                msg = "[ Epoch {} ] step {}/{} -loss:{:.4f}.".format(epoch, index, len(train_dataset), loss)
                reporter.console_log(msg, Fore.GREEN, save_db=False, show_task=False, show_step_sequence=False)

            if epoch % self.log_interval == 0:
                model_name = defense_model.__class__.__name__ + "(CIFAR-10)"
                file_name = "AT_" + "NATURAL" + '_' + model_name + "_" + str(epoch) + ".pt"
                save_weight_to_temp(model_name=model_name, defense_name="NATURAL", epoch_cursor=str(epoch), file_path=model_name + '/NATURAL/',
                                    file_name=file_name, weight=defense_model.state_dict())
            # val预测
            eval_test(val_dataset, defense_model, epoch, self.device)
            each_epoch_finish_callback(epoch)

        return defense_model.state_dict()

    def loss(self, model, x_natural, y, optimizer):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_natural), y)
        return loss

# self, lr=0.1, momentum=0.9, weight_decay=5e-4, epochs=10, run_device=None, log_interval=5
