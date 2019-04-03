import numpy as np
import torch

# For consistency represent vectors as D x 1

class MyForwardModel:
    def __init__(self, X=None, y=None, input_size=2, alpha=None, len_scale=1, noise_std=None):
        """
        X: N x D x 1
        y: N x S x 1
        Note S < D as ...
        """
        self.inv_sq_lens = torch.diag(
            1/(len_scale**2) * torch.ones(input_size)
        )

        self.alpha = alpha
        self.noise_std = noise_std

        self.X = X
        self.y = y

        self.K, self.L = self.calc_k_mat()

    def calc_k_mat(self):
        pass

    def _mult_by_K_inv(self, b):
        """
        b: N x C
        returns K^{-1}b : N x C
        This is for numerical stability
        """

        return torch.potrs(b, self.L, upper=False)

    def _kernel(self, x1, x2):
        """
        x1: (N) x D x 1
        x2: D x 1
        output: (N) x 1
        Note can be used to calculate k(X, x), however, to do K(X, X) you would need to loop over X for the first arg.
        Unless this is causing any speed issues no need to optimize!.
        """
        output = self.alpha**2 * torch.exp(-0.5 * torch.matmul(torch.transpose(
            x1-x2, dim0=-2, dim1=-1), torch.matmul(self.inv_sq_lens, x1-x2)))
        # output of size (N) x 1 x 1
        if x1 == x2:
            output += self.noise_std**2

        return output.squeeze(dim=-1)

    def mean(self, x):
        """
        x: D x 1
        output: S x 1
        """
        if not self.X:
            # Prior mean function
            return torch.zeros_like(x)
        
        output = torch.matmul(torch.t(self._kernel(X, x)), self._mult_by_K_inv(self.y))
        # output is 1 x S

        return torch.t(output) 

    def cov(self, x):
        """
        x: D x 1
        output: 1 x 1 TODO: Would need to consider case in which S > 1
        could perhaps assume the same K and hyperparameters across each S and that for f(x) = [f1(x), ..., fS(x)] there is no cross-correlation so that the covariance matrix is a diagonal matrix
        """
        kxx = self._kernel(x, x)
        if not self.X:
            return kxx
        
        return kxx - torch.matmul(torch.t(kxx), self._mult_by_K_inv(kxx))