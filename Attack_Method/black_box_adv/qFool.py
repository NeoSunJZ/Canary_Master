import os
import numpy as np
from copy import deepcopy
import torch
import torchvision.transforms as transforms
from scipy.fftpack import idct

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="qFool", perturbation_budget_var_name=None)
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="LocalSearchAttack",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT",
                                                          "selector": [{"value": "TARGETED", "name": "靶向"},
                                                                       {"value": "UNTARGETED", "name": "非靶向"}],
                                                          "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "max_queries": {"desc": "最大查询次数", "type": "INT", "def": "10000"},
                                          "tolerance": {"desc": "二分查找收敛阈值(距离< 此值)", "type": "FLOAT", "def": "1e-2"},
                                          "init_estimation": {"desc": "每次搜索初始查询次数", "type": "INT", "def": "100"},
                                          "eps": {"desc": "在某个点迭代的阈值", "type": "FLOAT", "def": "0.7"},
                                          # "omega0": {"desc": "调整预估梯度的参数的初始值（原论文公式4），保证正负例子比例约为55开", "type": "FLOAT"},
                                          # "phi0": {"desc": "调整预估梯度的参数的初始值（原论文公式4）保证正负例子比例约为55开", "type": "FLOAT"},
                                          "delta": {"desc": "攻击类型为TARGETED时有效，累加梯度的步长", "type": "FLOAT", "def": "10"},
                                          "subspacedim": {"desc": "DTC子空间的长宽（长宽一致）", "type": "INT", "def": "16"},
                                      })

class qFool():
    def __init__(self, model, run_device, attack_type='UNTARGETED', im_target=None, eps=0.7, clip_min=0, clip_max=1,
                 max_queries=10000, tolerance=1e-2, init_estimation=100, omega0=3 * 1e-2, delta=10, subspacedim=None, tlabel=-1):
        self.model = model
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attack_type = attack_type
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = eps

        self.tolerance = tolerance
        self.max_queries = max_queries
        self.init_estimation = init_estimation
        self.omega0 = omega0
        self.phi0 = -1
        self.delta = delta
        self.subspacedim = subspacedim
        self.tlable = tlabel
        self.n = []
        self.separatorline = "-" * 50
        if self.attack_type == 'TARGETED':
            self.target = im_target
            self.target_label = torch.from_numpy(np.array(self.tlabel)).to(self.device)

    @sefi_component.attack(name="qFool", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, imgs, ori_labels):
        self.image = imgs
        self.input_shape = self.image.cpu().numpy().shape
        input_dim = 1
        for s in self.input_shape:
            input_dim *= s
        self.input_dim = input_dim
        self.input_label = torch.from_numpy(np.array(ori_labels)).to(self.device)
        if not self.subspacedim:
             self.subspacedim = self.input_shape[2]

        if self.attack_type == 'UNTARGETED':
            r, n, label_orig, label_pert, pert_image = self.non_targeted()
        elif self.attack_type == 'TARGETED':
            r, n, label_orig, label_pert, pert_image = self.targeted()

        # xadv = self.inverse_transform(pert_image)
        # xpertinv = self.inverse_transform(r)

        # plt.figure()
        # plt.imshow(xadv)
        # plt.title(str_label_pert)
        # plt.figure()
        # plt.imshow(xpertinv)
        # plt.show()

        mse = torch.linalg.norm(r) / self.input_dim
        print("mse: ", mse)

        # return xadv, xpertinv, mse
        return pert_image


    def process_input_info(self):
        input_shape = self.image.numpy().shape
        input_dim = 1
        for s in input_shape:
            input_dim *= s
        input_label = self.query(self.image)
        return input_shape, input_dim, input_label


    def non_targeted(self):
        # line 1
        loop_i = 0

        # line 2
        P = []
        P0, num_calls1 = self.search_P0()
        P0, r, num_calls2 = self.bisect(self.image, P0)
        P.append(P0)

        print(self.separatorline)
        print("starting the loop to find x_adv ...")

        # line 3 and line 14
        x_adv, xi = [], []
        while np.sum(self.n) < self.max_queries:
            print(f"	#iteration: {loop_i}")

            # equation 4
            omega = self.omega0
            phi = self.phi0

            # line 4
            self.n.append(0)

            # line 5
            x_adv.append(P[loop_i])
            x_adv_prime = P[loop_i]

            # line 6 and line 12
            z_dot_eta = 0
            xi.append(0)
            while self.termination_condition(P[loop_i], self.n[loop_i], x_adv_prime, x_adv[loop_i]):
                # line 7
                z_dot_eta_new, rho, num_calls3 = self.estimate_gradient(P[loop_i], omega, phi)
                z_dot_eta += z_dot_eta_new
                print(f"     	#queries until now: n[{loop_i}]={self.n[loop_i] + num_calls3}")

                # line 8
                x_adv_prime = x_adv[loop_i]

                # line 9
                self.n[loop_i] += num_calls3  # n[loop_i] += self.init_estimation

                # line 10
                xi[loop_i] = self.normalize(z_dot_eta)

                # line 11
                x_adv[loop_i], num_calls4 = self.search_boundary_in_direction(self.image, xi[loop_i], r)
                x_adv[loop_i], r, num_calls5 = self.bisect(self.image, x_adv[loop_i])

                # equation 4
                omega, phi = self.update_omega(omega, phi, rho)

            print(f"	#queries: n[{loop_i}]={self.n[loop_i]}")

            # line 13
            P.append(x_adv[loop_i])
            loop_i += 1

        print("found x_adv!")

        # line 15
        pert_image = P[-1]
        r_tot = pert_image - self.image
        f_pert = self.query(pert_image)

        return r_tot, np.sum(self.n), self.input_label, f_pert, pert_image  # following deepfool's interface

    def targeted(self):
        # line 1
        loop_i = 0

        # line 2
        v = []
        v0 = self.normalize(self.target - self.image)
        v.append(v0)

        # line 3
        P = []
        P0, r, num_calls1 = self.bisect(self.image, self.target)
        P.append(P0)

        print(self.separatorline)
        print("starting the loop to find x_adv ...")

        # line 4 and line 17
        x_adv, xi, Q = [], [], []
        while np.sum(self.n) < self.max_queries:
            print(f"	#iteration: {loop_i}")

            # equation 4
            omega = self.omega0
            phi = self.phi0

            # line 5
            self.n.append(0)

            # line 6
            x_adv.append(P[loop_i])
            x_adv_prime = P[loop_i]

            # line 7 and line 15
            z_dot_eta = 0
            xi.append(0)
            Q.append(0)
            v.append(0)
            while self.termination_condition(P[loop_i], self.n[loop_i], x_adv_prime, x_adv[loop_i]):
                # line 8
                z_dot_eta_new, rho, num_calls2 = self.estimate_gradient(P[loop_i], omega, phi)
                z_dot_eta += z_dot_eta_new
                print(f"     	#queries until now: n[{loop_i}]={self.n[loop_i] + num_calls2}")

                # line 9
                x_adv_prime = x_adv[loop_i]

                # line 10
                self.n[loop_i] += num_calls2  # n[loop_i] += self.init_estimation

                # line 11
                xi[loop_i] = self.normalize(z_dot_eta)

                # line 12
                Q[loop_i], num_calls3 = self.search_boundary_in_direction(P[loop_i], xi[loop_i], self.delta)

                # line 13
                v[loop_i + 1] = self.normalize(Q[loop_i] - self.image)

                # line 14
                x_adv[loop_i], num_calls3 = self.search_boundary_in_direction(self.image, v[loop_i + 1], r)
                x_adv[loop_i], r, num_calls4 = self.bisect(self.image, x_adv[loop_i])

                # equation 4
                omega, phi = self.update_omega(omega, phi, rho)

            print(f"	#queries: n[{loop_i}]={self.n[loop_i]}")

            # line 16
            P.append(x_adv[loop_i])
            loop_i += 1

        print("found x_adv!")

        # line 15
        pert_image = P[-1]
        r_tot = pert_image - self.image
        f_pert = self.query(pert_image)

        return r_tot, np.sum(self.n), self.input_label, f_pert, pert_image  # following deepfool's interface

    def search_P0(self):
        P0 = deepcopy(self.image)
        sigma0, k = self.P0_constants()
        sigma = sigma0
        num_calls = 0

        print(self.separatorline)
        print("searching for the boundary using random noise ...")

        while not self.is_adversarial(P0) and num_calls < k:
            rj = self.create_perturbation() * sigma
            P0 = self.add_noise(self.image, rj)
            num_calls += 1
            sigma = sigma0 * (num_calls + 1)

            print(f"	sigma_{num_calls} = {sigma}", end='')
            print(f"	pert norm = {torch.linalg.norm(rj)}")
        # print(f"	pert mse = {1/self.input_dim*torch.linalg.norm(rj)}")

        if num_calls == k:
            print("cannot find P0!")
        else:
            print("found P0!")
        return P0, num_calls

    def P0_constants(self):
        sigma0 = 0.02
        k = 100
        return sigma0, k

    def termination_condition(self, Pi, ni, x_adv_prime, x_adv_i):
        cond1 = np.sum(self.n) < self.max_queries

        # note that "ni == 0" is redundant, since "x_adv_prime == Pi" includes that
        if torch.linalg.norm(x_adv_prime - Pi) == 0 or ni == 0:
            return cond1

        newrate = torch.linalg.norm(x_adv_i - Pi) / (ni + self.init_estimation)
        prevrate = torch.linalg.norm(x_adv_prime - Pi) / ni

        print(f"newrate/prevrate = {newrate / prevrate}")
        cond2 = newrate <= self.epsilon * prevrate
        return cond1 and cond2

    def estimate_gradient(self, Pi, omega, phi):
        z_dot_eta, cnt, num_calls = 0, 0, 0

        for _ in range(self.init_estimation):
            eta = self.create_perturbation()
            eta = self.normalize(eta) * omega
            P_eta = self.add_noise(Pi, eta)
            if self.is_adversarial(P_eta):
                z = 1
                cnt += 1
            else:
                z = -1
            z_dot_eta += z * eta
            num_calls += 1

        rho = 0.5 - cnt / self.init_estimation
        print(f"rho = {rho}")
        # num_calls = self.init_estimation
        return z_dot_eta, rho, num_calls

    def update_omega(self, omega, phi, rho):
        if rho > 0:
            new_phi = -phi
        else:
            new_phi = phi
        new_omega = omega * (1 + phi * rho)
        return new_omega, new_phi

    def bisect(self, image, adv_image):
        x = deepcopy(image)
        x_tilde = deepcopy(adv_image)
        num_calls = 0

        print("bisecting to get closer to boundary ...")
        while torch.linalg.norm(x - x_tilde) > self.tolerance:
            x_mid = (x + x_tilde) / 2
            print(f"	norm from image = {torch.linalg.norm(x_mid - image)}, ", end='')
            print(f"norm from adv = {torch.linalg.norm(x_mid - adv_image)}")

            if self.is_adversarial(x_mid):
                x_tilde = x_mid
            else:
                x = x_mid
            num_calls += 1
        print("bisection done!")
        return x_tilde, torch.linalg.norm(x_tilde - image), num_calls

    def create_perturbation(self):
        pert_tensor = torch.zeros_like(self.image)

        # added after implementing subspaces
        subspace_shape = (1, self.input_shape[0], self.subspacedim, self.subspacedim)
        pert_tensor[:, :, :self.subspacedim, :self.subspacedim] = torch.randn(size=subspace_shape)

        if self.subspacedim < self.input_shape[1]:
            pert_tensor = torch.from_numpy(idct(idct(pert_tensor.numpy(), axis=3, norm='ortho'),
                                                axis=2, norm='ortho'))

        return pert_tensor

    def search_boundary_in_direction(self, image, xi, r):
        x_adv = self.add_noise(image, xi * r)
        num_calls = 1
        while not self.is_adversarial(x_adv):
            num_calls += 1
            x_adv = self.add_noise(image, xi * num_calls * r)
        return x_adv, num_calls

    def add_noise(self, image, pert):
        x_tilde = image + pert
        return self.clip_tensor(x_tilde)

    def normalize(self, vec):
        return vec / torch.linalg.norm(vec)

    def query(self, image):
        x = deepcopy(image)
        if len(x) == 3:
            x = x[None, :]
        output = self.model(x)
        top1_label = output.max(1, keepdim=True)[1]
        return top1_label

    def is_adversarial(self, x):
        top1_label = self.query(x)
        if self.attack_type == 'UNTARGETED':
            return not top1_label.item() == self.input_label.item()
        elif self.attack_type == 'TARGETED':
            return top1_label.item() == self.target_label.item()

    def bounds(self):
        zeros = transforms.ToPILImage()(torch.zeros_like(self.image))
        fulls = transforms.ToPILImage()(torch.ones_like(self.image))

        lb_tensor = self.transform_input(zeros)
        ub_tensor = self.transform_input(fulls)
        return lb_tensor, ub_tensor

    def clip_tensor(self, x):
        # xc = torch.maximum(x, self.clip_min)
        # xc = torch.minimum(xc, self.clip_max)
        # xc = torch.maximum(x, torch.zeros_like(self.image))
        # xc = torch.minimum(xc, 255*torch.ones_like(self.image))
        return torch.clamp(x, self.clip_min, self.clip_max)

