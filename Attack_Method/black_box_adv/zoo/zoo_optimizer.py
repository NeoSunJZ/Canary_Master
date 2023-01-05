import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class ZOOptim(object):

    def __init__(self, model, loss, device='cuda'):
        """
        Args:
            Name            Type                Description
            model:          (nn.Module)         The model to use to get the output
            loss:           (nn.Module)         The loss to minimize
            device:         (str)               Used device ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.loss = loss
        self.model = model.to(self.device)
        self.model.eval()


    """
    Perform a zero-order optimization
    """
    def run(self, x, c, learning_rate=1e-2, n_gradient=128,
            h=1e-4, beta_1=0.9, beta_2=0.999, solver="adam", verbose=False,
            max_steps=10000, batch_size=-1, C=(0, 1),  stop_criterion=True,
            tqdm_disabled=False, additional_out=False):
        """
        Args:
            Name                    Type                Description
            x:                      (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
            c:                      (float)             Confidence
            learning_rate:          (float)             Learning rate
            n_gradient:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Gradient estimation accuracy O(h^2)
            beta_1:                 (float)             ADAM hyper-parameter
            beta_2:                 (float)             ADAM hyper-parameter
            solver:                 (str)               Either "ADAM" or "Newton"
            verbose:                (int)               Display information. Default is 0
            max_steps:              (int)               The maximum number of steps
            batch_size:             (int)               Maximum number of parallelization during model call
            C:                      (tuple)             The boundaries of a pixel
            stop_criterion:         (boolean)           If true stop when the loss does not decrease for 20 iterations
            tqdm_disabled:          (bool)              Disable the tqdm bar. Default is False
            additional_out:         (bool)              Return also all the x. Default is False
        return:
            x:                      (torch.tensor)      Final image
            losses:                 (list)              Loss at each step
            outs:                   (list)              Adversarial example at each step
            l2_dists:               (list)              L2-dist at each step
            losses_st:              (list)              Second loss term at each step
        """

        # Verify Solver is implemented
        if solver.lower() not in ["adam", "newton"]:
            raise NotImplementedError("Unknown solver, use 'adam' or 'newton'")

        # 1. Initialize paramters
        x = x.to(self.device)
        total_dim = int(np.prod(x.shape))
        x_dim = x.shape
        # Reshape x to be column vector
        x = x.reshape(-1, 1)
        # Store original image
        x_0 = x.clone().detach()
        # Init list of losses (made up of two terms) and outputs for results
        losses, losses_st, l2_dists, outs = [], [], [], []
        # Best Values
        best_l2 = np.inf
        best_image = torch.zeros_like(x)
        best_loss = np.inf
        # Adversarial attack found flag
        found = False

        # Set ADAM parameters
        if solver == "adam":
            self.M = torch.zeros(x.shape).to(self.device)
            self.v = torch.zeros(x.shape).to(self.device)
            self.T = torch.zeros(x.shape).to(self.device)

        # 2. Main Iteration
        for iteration in tqdm(range(max_steps), disable=tqdm_disabled):

            # 2.1 Call the step
            x = self.step(x, x_0, c, learning_rate, n_gradient, h, beta_1, beta_2,
                          solver, x_dim, total_dim, batch_size, C, verbose).detach()

            # 2.2 Compute new loss and store current info
            out = self.model(x.view(1, x_dim[0], x_dim[1], x_dim[2]))
            curr_loss_st = self.loss(out)
            curr_l2 = torch.norm(x - x_0).detach()
            curr_loss = curr_l2 + c * curr_loss_st

            # 2.3 Flag when a valid adversarial example is found
            if curr_loss_st == 0:
                found = True

            # 2.4 Keep best values
            if found:
                # First valid example
                if not losses_st or losses_st[-1] != 0:
                    if not tqdm_disabled:
                        print("First valid image found at iteration {} with l2-distance = {}".format(iteration, curr_l2))
                    best_l2 = curr_l2
                    best_image = x.detach().clone()
                    best_loss = curr_loss
                # New best example
                if curr_l2 < best_l2 and curr_loss_st == 0:
                    best_l2 = curr_l2
                    best_image = x.detach().clone()
                    best_loss = curr_loss
                # Worst example
                else:
                    curr_l2 = best_l2
                    curr_loss = best_loss
                    x = best_image.clone()
                    curr_loss_st = torch.zeros(1)

            # 2.5 Save results
            outs.append(float(out.detach().cpu()[0, self.loss.neuron].item()))
            losses_st.append(float(curr_loss_st.detach().cpu().item()))
            l2_dists.append(curr_l2)
            losses.append(curr_loss.detach())

            # 2.6 Display  info
            if verbose:
                print('-----------STEP {}-----------'.format(iteration))
                print('Iteration: {}'.format(iteration))
                print('Shape of x: {}'.format(x.shape))
                print('Second term loss:    {}'.format(curr_loss_st.cpu().item()))
                print('L2 distance: {}'.format(curr_l2))
                print("Adversarial Loss:    {}".format(curr_loss))

            # 2.7 Evaluate stop criterion
            if stop_criterion:
              if len(losses) > 20:
                if curr_loss_st == 0 and l2_dists[-1] == l2_dists[-20]:
                    break

        # 3. Unsuccessful attack
        if losses_st[-1] > 0:
            print("Unsuccessful attack")
            return x.reshape(x_dim[0], x_dim[1], x_dim[2]), losses, outs

        # Return
        best_image = best_image.reshape(x_dim[0], x_dim[1], x_dim[2])
        if additional_out:
            return best_image, losses, l2_dists, losses_st, outs
        return best_image, losses, outs


    """
    Do an optimization step
    """
    def step(self, x, x_0, c, learning_rate, n_gradient, h, beta_1, beta_2,
             solver, x_dim, total_dim, batch_size, C, verbose):
        """
        Args:
            Name                    Type                Description
            x:                      (torch.tensor)      Linearized current adversarial image
            x_0:                    (torch.tensor)      Linearized original image
            c:                      (float)             Regularization parameter
            learning_rate:          (float)             Learning rate
            n_gradient:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Gradient estimation accuracy O(h^2)
            beta_1:                 (float)             ADAM hyper-parameter
            beta_2:                 (float)             ADAM hyper-parameter
            solver:                 (str)               Either "ADAM" or "Newton"
            x_dim:                  (torch.size)        Size of the original image
            total_dim:              (int)               Total number of pixels
            batch_size:             (int)               Maximum number of parallelization during model call
            C:                      (tuple)             The boundaries of a pixel
            verbose:                (int)               Display information. Default is 0
        return:
            x:                      (torch.tensor)      Adversarial example
        """

        # 1. Select n_gradient dimensions
        if n_gradient > total_dim:
            raise ValueError("Batch size must be lower than the total dimension")

        indices = np.random.choice(total_dim, n_gradient, replace=False)
        e_matrix = torch.zeros(n_gradient, total_dim).to(self.device)
        for n, i in enumerate(indices):
            e_matrix[n, i] = 1

        # 2. Optimizers
        if solver == "adam":
            # 2.1 Gradient approximation
            g_hat = self.compute_gradient(x_0, x, e_matrix, c, n_gradient, h, solver, x_dim, total_dim, batch_size, verbose).view(-1, 1)

            # 2.2 ADAM Update
            self.T[indices] = self.T[indices] + 1
            self.M[indices] = beta_1 * self.M[indices] + (1 - beta_1) * g_hat
            self.v[indices] = beta_2 * self.v[indices] + (1 - beta_2) * g_hat ** 2
            M_hat = torch.zeros_like(self.M).to(self.device)
            v_hat = torch.zeros_like(self.v).to(self.device)
            M_hat[indices] = self.M[indices] / (1 - beta_1 ** self.T[indices])
            v_hat[indices] = self.v[indices] / (1 - beta_2 ** self.T[indices])
            delta = -learning_rate * (M_hat / (torch.sqrt(v_hat) + 1e-8))
            x = x + delta.view(-1, 1)

            # 2.3 Call verbose
            if verbose > 1:
                print('-------------ADAM------------')
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('Chosen indices are: {}\n'.format(indices))
                print('T = {}'.format(self.T))
                print('M = {}'.format(self.M))
                print('v = {}'.format(self.v))
                print('delta = {}'.format(delta))

            # 2.4 Return
            g_hat.detach(), M_hat.detach(), v_hat.detach(), delta.detach()
            return self.project_boundaries(x, C).detach()

        elif solver == "newton":
            # 2.1 Gradient and Hessian approximation
            g_hat, h_hat = self.compute_gradient(x_0, x, e_matrix, c, n_gradient, h, solver, x_dim, total_dim, batch_size, verbose)
            g_hat = g_hat.view(-1, 1)
            h_hat = h_hat.view(-1, 1)

            # 2.2 Update
            delta = torch.zeros(x.shape).to(self.device)
            h_hat[h_hat <= 0] = 1
            h_hat[h_hat <= 1e-3] = 1e-3
            delta[indices] = -learning_rate * (g_hat / h_hat)
            x = x + delta.view(-1, 1)

            # 2.3 Call verbose
            if verbose > 1:
                print('------------NEWTON-----------')
                print('Chosen indices are: {}\n'.format(indices))
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('delta = {}'.format(delta))

            # 2.4 Return
            g_hat.detach(), h_hat.detach(), delta.detach()
            return self.project_boundaries(x, C).detach()


    """
    Compute Gradient and Hessian
    """
    def compute_gradient(self, x_0, x, e_matrix, c, n_gradient, h, solver, x_dim, total_dim, batch_size, verbose):
        """
        Args:
            x_0:                    (torch.tensor)      Linearized original image
            x:                      (torch.tensor)      Linearized current adversarial image
            e_matrix:
            c:                      (float)             Regularization parameter
            n_gradient:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Gradient estimation accuracy O(h^2)
            solver:                 (str)               Either "ADAM" or "Newton"
            x_dim:                  (torch.size)        Size of the original image
            total_dim:              (int)               Total number of pixels
            batch_size:             (int)               Maximum number of parallelization during model call
            verbose:                (int)               Display information. Default is 0
        return:
            g_hat:                  (torch.tensor)      Gradient approximation
            h_hat:                  (torch.tensor)      Hessian approximation (if solver=="Newton")
        """

        # 1. Initialize useful parameters
        if batch_size == -1:
            batch_size = n_gradient

        n_batches = n_gradient//batch_size
        g_hat = torch.zeros(n_gradient, 1).to(self.device)
        h_hat = torch.zeros(n_gradient, 1).to(self.device)

        # 2. Cycle through each batch to compute gradient and hessian approximation
        for i in range(n_batches):
            # 2.1 Expand image along one direction
            x_expanded = x.view(-1).expand(batch_size, total_dim).to(self.device)
            x_0_expanded = x_0.view(-1).expand(batch_size, total_dim).to(self.device)

            # 2.2 Intermediate steps
            input_plus = x_expanded + h * e_matrix[batch_size*i:batch_size*(i+1), :]
            out_plus = self.model(input_plus.view(batch_size, *list(x_dim)))
            loss2_plus = self.loss(out_plus)
            loss1_plus = torch.norm(input_plus - x_0_expanded, dim=1)
            input_plus.detach()
            input_minus = x_expanded - h * e_matrix[batch_size*i:batch_size*(i+1), :]
            out_minus = self.model(input_minus.view(batch_size, *list(x_dim)))
            loss2_minus = self.loss(out_minus)
            loss1_minus = torch.norm(input_minus - x_0_expanded, dim=1)
            input_minus.detach()
            first_term = loss1_plus + (c * loss2_plus)
            second_term = loss1_minus + (c * loss2_minus)

            # 2.3. Compute gradient
            g_hat[batch_size * i:batch_size * (i + 1), :] = (first_term.view(-1, 1) - second_term.view(-1, 1)) / (2 * h)

            # 2.4 Compute Hessian if we are using Newton's method
            if solver == 'newton':
                loss1_added = torch.norm(x_expanded - x_0_expanded, dim=1)
                loss2_added = self.loss(self.model(x_expanded.view(batch_size, *list(x_dim))))
                additional_term = loss1_added + (c * loss2_added)
                h_hat[batch_size * i:batch_size * (i + 1), :] = (first_term.view(-1, 1) + second_term.view(-1, 1) - 2 * additional_term.view(-1, 1)) / (h ** 2)

        # 3. Display info
        if verbose > 0:
            print('-----COMPUTING GRADIENT-----')
            print('g_hat has shape {}'.format(g_hat.shape))
            print('g_hat ='.format(g_hat))
            if solver == 'newton':
                print('h_hat has shape {}'.format(h_hat.shape))
                print('h_hat ='.format(h_hat))

        # 4. Return
        if solver == 'newton':
            return g_hat.detach(), h_hat.detach()
        else:
            return g_hat.detach()


    """
    Check the boundaries of our constraint optimization problem
    """
    def project_boundaries(self, x, C):
        """
        Args:
            Name              Type                Description
            x:                (torch.tensor)      Linearized current adversarial image
            C:                (tuple)             The boundaries of a pixel
        return:
            x:                (torch.tensor)      Linearized current adversarial image
        """

        x[x > C[1]] = C[1]
        x[x < C[0]] = C[0]
        return x

