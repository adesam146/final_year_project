import numpy as np
import torch
import gpytorch

# Planning to use gpytorch since it integrates well
# with pytorch, is modular (so should be quite flexible) and
# as the advantage of making use of GPUs if needed.


class ForwardModel:
    pass


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """
        Let S be the dimension of the functions output,
        D the dimension of the input and N the number of data.
        We would model the actual function with S separate/independant GPs 
        train_y: S x N x 1
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
