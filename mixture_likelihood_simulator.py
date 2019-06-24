import torch
import torch.distributions as trd
import numpy as np
import scipy
from matplotlib import pyplot as plt

class MixtureLikelihoodSimulator():
    def __init__(self, noise_std, convert_to_design_matrix):
        self.noise_std = noise_std

        # This assumes the convertion function retains the shape of X as
        # N x 1
        self.convert_to_design_matrix = convert_to_design_matrix

    def simulate(self, beta, X):
        """
        X should be of the form (N, 1)
        beta is of the form (1, 1)
        output has shape (N, 1)
        """
        N = X.shape[0]
        Y = torch.zeros(N)

        pos_idx =  (X > 1) * (X < 1.25) + (X > 1.75) * (X < 2)
        mean = torch.matmul(self.convert_to_design_matrix(X), beta.view(-1,1))
        Y = mean + self.noise_std * torch.randn(N, 1)
        Y[pos_idx] = trd.Uniform(low=-4*mean[pos_idx], high=4*mean[pos_idx]).sample()

        return Y


