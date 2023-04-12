import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from canary_lib.canary_attack_method.black_box_adv.tremba.fcn import Imagenet_Encoder, Imagenet_Decoder
from canary_lib.canary_attack_method.black_box_adv.tremba.utils import MarginLoss


def train_generator(run_device, nets, net_name_list, dataset, batch_size, epochs,
                    learning_rate_G, momentum_G, schedule_G, gamma_G, margin, is_target, target_class, epsilon,
                    weight_save_path, weight_save_name):
    model = nn.Sequential(
        Imagenet_Encoder(),
        Imagenet_Decoder()
    )
    model.to(run_device)

    optimizer_G = torch.optim.SGD(model.parameters(), learning_rate_G, momentum=momentum_G, weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=epochs // schedule_G, gamma=gamma_G)
    hingeloss = MarginLoss(margin=margin, target=is_target)

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def train():
        model.train()
        for batch_idx, (data, label) in enumerate(dataset_loader):

            nat = data.to(run_device)
            if is_target:
                label = target_class
            else:
                label = label.to(run_device)

            losses_g = []
            optimizer_G.zero_grad()
            for net in nets:
                noise = model(nat)
                adv = torch.clamp(noise * epsilon + nat, 0, 1)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()

            if (batch_idx + 1) % 100 == 0:
                print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(net_name_list, losses_g))))


    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]

        for batch_idx, (data, label) in enumerate(dataset_loader):

            nat = data.to(run_device)
            if is_target:
                label = target_class
            else:
                label = label.to(run_device)
            noise = model(nat)
            adv = torch.clamp(noise * epsilon + nat, 0, 1)

            for j in range(len(nets)):
                logits = nets[j](adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                if is_target:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1) != label).sum())

        test_loss = [loss_avg[i] / len(dataset_loader) for i in range(len(loss_avg))]
        test_successes = [success[i] / len(dataset_loader.dataset) for i in range(len(success))]
        test_success = 0.0
        for i in range(len(test_successes)):
            test_success += test_successes[i] / len(test_successes)

        return test_success

    for epoch in tqdm(range(epochs)):
        scheduler_G.step()
        train()
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            with torch.no_grad():
                test_success = test()
                print("epoch {}, Current success: {}".format(epoch, test_success))
        if not os.path.exists(weight_save_path):
            os.makedirs(weight_save_path)

        torch.save(model.state_dict(), os.path.join(weight_save_path, weight_save_name))
