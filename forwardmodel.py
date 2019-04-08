import numpy as np
import torch
import gpytorch
import torch.distributions as trd

# Planning to use gpytorch since it integrates well
# with pytorch, is modular (so should be quite flexible) and
# as the advantage of making use of GPUs if needed.


class ForwardModel:
    def __init__(self, init_x, init_y):
        """
        This model assumes the states and actions have already been paired
        to form the input into the GP
        init_x: N x D
        init_y: N x S
        """
        S = init_y.shape[1]
        # train_x is now S x N x D which is what the GP expects
        self.train_x = init_x.unsqueeze(0).repeat(S, 1, 1)
        # train_y is now S x N
        self.train_y = torch.t(init_y)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=S)
        self.model = GPModel(self.train_x, self.train_y, self.likelihood)

    def rsample(self, x):
        """
        x: (Tst) x D
        output: S
        """
        self.model.eval()
        self.likelihood.eval()

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # We squeeze since output without would be: n x S x Tst = 1 x 1 x 1 in this case. Where n is number of sample and Tst is number of test points
        return self.likelihood(self.model(x.view(-1, x.shape[-1]))).rsample().squeeze()

    def mean(self, x):
        """
        x: (Tst) x D
        output: S
        """
        self.model.eval()
        self.likelihood.eval()

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # We squeeze since output without would be: n x S x Tst = 1 x 1 x 1 in this case. Where n is number of sample and Tst is number of test points
        return self.likelihood(self.model(x.view(-1, x.shape[-1]))).mean.squeeze()


    def train(self):
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """
        Let S be the dimension of the functions output,
        D the dimension of the input and N the number of data.
        We would model the actual function with S separate/independant GPs 
        train_y: S x N
        train_x: S x N x D
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        num_output = train_y.shape[0]
        self.mean_module = gpytorch.means.ConstantMean(batch_size=num_output)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=num_output), batch_size=num_output)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
