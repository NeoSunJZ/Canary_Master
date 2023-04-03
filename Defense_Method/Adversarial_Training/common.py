import torch
import torch.nn.functional as F
from colorama import Fore

from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


def eval_test(val_dataset, defense_model, epoch, device):
    acc = 0
    val_loss = 0
    for index in range(len(val_dataset)):
        defense_model.eval()
        data, target = val_dataset[index][0].to(device), val_dataset[index][1].to(device)
        logit = defense_model(data)
        val_loss += F.cross_entropy(logit, target, size_average=False).item()
        output = F.softmax(logit, dim=1)
        pred_label = torch.argmax(output, dim=1)
        acc += torch.sum(torch.eq(pred_label, target))

    msg = "[ Val ] epoch {} -val_loss:{:.4f} -acc:{:.4f}.".format(epoch, val_loss / val_dataset.dataset_size,
                                                                  acc / val_dataset.dataset_size)
    reporter.console_log(msg, Fore.GREEN, show_task=False, show_step_sequence=False)


def adjust_learning_rate(ori_lr, optimizer, epoch):
    """decrease the learning rate"""
    lr = ori_lr
    if epoch >= 15:
        lr = ori_lr * 0.1
    if epoch >= 90:
        lr = ori_lr * 0.01
    if epoch >= 100:
        lr = ori_lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
