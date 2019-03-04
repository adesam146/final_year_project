import numpy as np
import torch
import torch.distributions as trd
from torch.distributions import transform_to
from torch.distributions.transforms import LowerCholeskyTransform

# NOTE: Not storing the distributions as a field because parameters
# such as variance need to be constrained and if this was done in the constructor autograd would try to 'backprog' to the 'Trainable*' constructor which is would have disposable of after one iteration of optimisation

#TODO: when API is more settled and common patterns appear extract a abstract parent class

class TrainableNormal:
    def __init__(self, init_mean=0, init_std_dev=1):
        self.mean = torch.tensor(
            init_mean, requires_grad=True, dtype=torch.float)
        self.ln_sigma = torch.tensor(
            np.log(init_std_dev), requires_grad=True, dtype=torch.float)

    def log_prob(self, x):
        return trd.Normal(loc=self.mean, scale=torch.exp(self.ln_sigma)).log_prob(x)

    def rsample(self, sample_size=1):
        return trd.Normal(loc=self.mean, scale=torch.exp(self.ln_sigma)).rsample((sample_size, 1))

    def sample(self, sample_size=1):
        return trd.Normal(loc=self.mean, scale=torch.exp(self.ln_sigma)).sample((sample_size, 1))

    def parameters(self):
        return [self.mean, self.ln_sigma]

    def eval(self):
        self.mean.requires_grad = self.ln_sigma.requires_grad = False

    def train(self):
        self.mean.requires_grad = self.ln_sigma.requires_grad = True