import numpy as np
import torch
import gpytorch
import torch.distributions as trd
from matplotlib import pyplot as plt
import matplotlib.colorbar as cbar

# Planning to use gpytorch since it integrates well
# with pytorch, is modular (so should be quite flexible) and
# as the advantage of making use of GPUs if needed.


class ForwardModel:
    def __init__(self, init_x, init_y, D, S, F, device, save_dir=''):
        """
        This model assumes the states and actions have already been paired
        to form the input into the GP
        init_x: N x S+F
        init_y: N x D
        """
        assert init_x.shape[-1] == S+F
        assert init_y.shape[-1] == D
        self.D = D
        self.S = S
        self.F = F
        self.device = device
        self.save_dir = save_dir
        self.train_x_filename = 'train_x.pt'
        self.train_y_filename = 'train_y.pt'

        self.train_x, self.train_y = self.__process_inputs(init_x, init_y)

        # See if this can be uncommented, I ideally want the same likelhood module throughout as I feel it allows reuse info?? Thereby, making finding optimal hyper-param easier
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=D)
        self.likelihood = None
        self.model = None

    def predict(self, x, with_rsample=True, return_log_prob=False):
        """
        x: 1 x S+F
        sample: 1 x D
        log_prob: 1
        """
        assert self.model is not None
        assert x.shape[0] == 1
        assert x.shape[1] == self.S + self.F

        dyn_model = self.predictive_distn(x)

        if with_rsample:
            sample = dyn_model.rsample()
        else:
            sample = dyn_model.sample()

        log_prob = None
        if return_log_prob:
            log_prob = dyn_model.log_prob(sample).sum()

        return torch.t(sample), log_prob

    def predictive_distn(self, x):
        """
        x: 1 x S+F
        output: A gpytorch.MultivariateNormal with mean D x 1
        """
        assert self.model is not None
        assert x.shape[0] == 1
        assert x.shape[1] == self.S + self.F

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # Output from sample is (n x ) D x Tst = (1 x) 1 x 1. Where n is number of sample and Tst is number of test points
        dyn_model = self.model(self.__adjust_shape(x))

        return dyn_model

    def learn(self):
        if self.model is not None:
            # Consider how model can be set in constructor and reused
            # i.e. the fantasy model thing
            self.model.cpu()
            self.likelihood.cpu()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_size=self.D).to(self.device)
        self.model = GPModel(self.train_x, self.train_y,
                             self.likelihood).to(self.device)

        print("***BEFORE OPTIMATION***")
        self.model.covar_module.outputscale = torch.var(
            self.train_y, dim=1).squeeze()

        kappa = 2
        self.model.likelihood.noise = torch.var(
            self.train_y, dim=1).squeeze() / (kappa**2)

        lambda_sq = 3**2
        self.model.covar_module.base_kernel.lengthscale = torch.var(
            self.train_x[0], dim=0, keepdim=True).unsqueeze(0).repeat(self.D, 1, 1) / lambda_sq

        self.mll_optimising_progress()

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            # Includes GaussianLikelihood parameters
            {'params': self.model.parameters()},
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)

        training_iter = 2000
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            if i < 20 or i > training_iter - 21:
                # Only print first 20 and last 20 losses
                print('Iter %d/%d - Loss: %.3f' %
                  (i + 1, training_iter, loss.item()))
            optimizer.step()

        self.model.freeze_parameters()
        self.model.eval()
        self.likelihood.eval()

        print("***AFTER OPTIMATION***")
        self.mll_optimising_progress()

        # Saving current training data so it can be recovered after
        # temporary data points are added during trajectory prediction
        self.save_training_data()
        self.save_model_state()

        # Random computation for force the kernel to be evaluated
        # The reason this is done is that you do not want the first time the kernel for the data is calculated to be in a loop and involve a tensor that requires_grad because onces backward is called, in a subsequence iteration due to gpytorch's lazy loading, autograd would try to access the TODO
        self.predict(torch.zeros(1, self.S+self.F, device=self.device))

    def mll_optimising_progress(self):
        print("Noise Variance:", self.model.likelihood.noise)
        print("Lengthscale:", self.model.covar_module.base_kernel.lengthscale)
        print("Signal Variance:", self.model.covar_module.outputscale)

        print("Estimated target variance:", torch.var(self.train_y, dim=1))

        signal_to_noise = torch.sqrt(
            self.model.covar_module.outputscale/self.model.likelihood.noise.squeeze())
        N = self.train_y.shape[1]
        print("N:", N)
        print("Signal to noise ratio:", signal_to_noise)
        print("Bound on condition number:", N * signal_to_noise**2 + 1)

    def update_data(self, new_x, new_y):
        """
        new_x: N x S+F
        new_y: N x D
        """
        assert new_x.shape[-1] == self.S + self.F
        assert new_y.shape[-1] == self.D

        new_x, new_y = self.__process_inputs(new_x, new_y)

        self.train_x = torch.cat((self.train_x, new_x), dim=1)
        self.train_y = torch.cat((self.train_y, new_y), dim=1)

    def add_fantasy_data(self, x, y):
        """
        x: (N x) S+F
        y: (N x) D
        """
        assert self.model is not None
        assert x.shape[-1] == self.S+self.F
        assert y.shape[-1] == self.D

        x, y = self.__process_inputs(
            x.view(-1, self.S+self.F), y.view(-1, self.D))

        y = y + torch.sqrt(self.model.likelihood.noise) * torch.randn_like(y)

        self.train_x = torch.cat((self.train_x, x), dim=1)
        self.train_y = torch.cat((self.train_y, y), dim=1)

        self.model.set_train_data(self.train_x, self.train_y, strict=False)

    def clear_fantasy_data(self):
        del self.train_x, self.train_y

        self.train_x = torch.load(self.save_dir + self.train_x_filename)
        self.train_y = torch.load(self.save_dir + self.train_y_filename)

        self.model.set_train_data(self.train_x, self.train_y, strict=False)

    def save_training_data(self):
        torch.save(self.train_x, self.save_dir + self.train_x_filename)
        torch.save(self.train_y, self.save_dir + self.train_y_filename)

    def save_model_state(self):
        torch.save(self.model.state_dict(),
                   self.save_dir + 'gp_model_state.pt')

    def __process_inputs(self, x, y):
        """
        Puts inputs in batch form for the Batch GP
        x: N x S+F
        y: N x D
        output[0]: D x N x S+F
        output[1]: D x N
        """
        x = self.__adjust_shape(x)
        # x is now D x N x S+F which is what the GP expects
        y = torch.t(y)
        # y is now D x N

        return x, y

    def __adjust_shape(self, x):
        """
        x: Tst x S+F
        output: D x Tst x S+F
        Need to replicate x for each target dimension
        """
        return x.unsqueeze(0).expand(self.D, -1, -1)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """
        Let D be the dimension of the functions output,
        S+F the dimension of the input and N the number of data.
        We would model the actual function with D separate/independant GPs 
        train_y: D x N
        train_x: D x N x S+F
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        num_output = train_y.shape[0]
        input_dim = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean(batch_size=num_output)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=num_output, ard_num_dims=input_dim), batch_size=num_output)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def freeze_parameters(self):
        """
        This would also prevent grads from the likelihood parameters
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """
        This would also allow grads from the likelihood parameters
        """
        for param in self.parameters():
            param.requires_grad = True
