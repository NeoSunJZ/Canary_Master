import argparse
import torch
import numpy as np
from foolbox import PyTorchModel
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType
import canary_lib.canary_attack_method.black_box_adv.triangle_attack.attack_mask as attack
sefi_component = SEFIComponent()
@sefi_component.attacker_class(attack_name="TA")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="TA",
                                      handler_type=ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })

class TA:
    def __init__(self, model, run_device, attack_type):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--max_queries',
            type=int,
            default=10000,
            help='The max number of queries in model'
        )
        parser.add_argument(
            '--ratio_mask',
            type=float,
            default=0.1,
            help='ratio of mask'
        )
        parser.add_argument(
            '--dim_num',
            type=int,
            default=1,
            help='the number of picked dimensions'
        )
        parser.add_argument(
            '--max_iter_num_in_2d',
            type=int,
            default=2,
            help='the maximum iteration number of attack algorithm in 2d subspace'
        )
        parser.add_argument(
            '--init_theta',
            type=int,
            default=2,
            help='the initial angle of a subspace=init_theta*np.pi/32'
        )
        parser.add_argument(
            '--init_alpha',
            type=float,
            default=np.pi / 2,
            help='the initial angle of alpha'
        )
        parser.add_argument(
            '--plus_learning_rate',
            type=float,
            default=0.1,
            help='plus learning_rate when success'
        )
        parser.add_argument(
            '--minus_learning_rate',
            type=float,
            default=0.005,
            help='minus learning_rate when fail'
        )
        parser.add_argument(
            '--half_range',
            type=float,
            default=0.1,
            help='half range of alpha from pi/2'
        )
        self.args = parser.parse_args()
        self.args.side_length = 224
        # self.model = model
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        if torch.cuda.is_available():
            mean = mean.to(self.device)
            std = std.to(self.device)
        preprocessing = dict(mean=mean, std=std, axis=-3)
        self.model = PyTorchModel(model, bounds=(0, 1), device=run_device)

    @sefi_component.attack(name="TA", is_inclass=True, support_model=[], attack_type="WHITE_BOX")
    def attack(self, imgs, ori_labels, tlabels=None):
        # ori_labels = ep.astensor(torch.from_numpy(np.array(ori_labels)).to(self.device))
        # self.init_attack: MinimizationAttack
        # self.init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
        # criterion = Misclassification(labels=ori_labels)
        # criterion = get_criterion(criterion)
        # imgs = ep.astensor(imgs)
        # originals, restore_type = ep.astensor_(imgs)
        # best_advs = self.init_attack.run(
        #     self.model, originals, criterion, early_stop=None
        # )
        # return restore_type(best_advs).raw

        ori_labels = np.array(ori_labels)
        ori_labels = torch.from_numpy(ori_labels).to(self.device).long()
        ta_model = attack.TA(self.model, input_device=self.device)
        my_advs, q_list, my_intermediates, max_length = ta_model.attack(self.args, imgs, ori_labels)
        return my_advs