import torch
import torch.nn as nn


class RBFPolicy(nn.Module):
    # Consider making an nn module
    def __init__(self, u_max, input_dim, nbasis, device):
        assert input_dim == 5  # Only dealing with the cartpole problem for now
        super().__init__()
        self.u_max = u_max
        self.input_dim = input_dim
        self.nbasis = nbasis
        self.device = device

        # Most of the initialisations here are based on the PILCO code

        self.X = nn.Parameter(torch.randn(
            self.nbasis, self.input_dim, device=self.device))

        self.log_l = nn.Parameter(
            torch.log(torch.tensor([1, 1, 1, 0.7, 0.7], device=self.device)))

        self.Y = nn.Parameter(
            0.1 * torch.randn(self.nbasis, device=self.device))

    def __call__(self, x):
        self.recompute_K()

        k = torch.zeros(self.nbasis, device=self.device)

        for i in range(self.nbasis):
            k[i] = self.kernel(x, self.X[i])

        l_inv = torch.cholesky(self.K).inverse()

        v = torch.chain_matmul(l_inv.t(), l_inv, self.Y.view(-1, 1))

        return self.u_max * self.squash(torch.matmul(k.view(1, -1), v)).view(1)

    def recompute_K(self):
        """
        Signal variance is implictly set to 1 and
        Signal noise variance to 0.01**2
        """
        self.K = torch.diag(self.__k(self.X.unsqueeze(1) - self.X)).view(self.nbasis, self.nbasis) + 0.01**2 * torch.eye(self.nbasis)

        # self.K2 = torch.zeros(self.nbasis, self.nbasis, device=self.device)
        # for i in range(self.nbasis):
        #     for j in range(self.nbasis):
        #         self.K2[i, j] = self.kernel(self.X[i], self.X[j])
        #         if i == j:
        #             self.K2[i, j] += 0.01**2

    def kernel(self, x1, x2):
        return self.__k(x1-x2)

    def __k(self, diff):
        """
        diff: (N x) input_dim
        output: N X N
        """
        diff = diff.view(-1, self.input_dim)
        return torch.exp(-0.5 * torch.chain_matmul(diff, torch.diag(torch.exp(-2 * self.log_l)), diff.t()))

    def squash(self, x):
        """
        Squashing the values in x to be between -1 and 1
        """
        return (9*torch.sin(x) + torch.sin(3*x))/8

    def eval(self):
        pass

    def train(self):
        pass

# Testing Policy, TODO: Convert to test
# policy = RBFPolicy(1, input_dim=1, nbasis=2, device=torch.device('cpu'))

# print("Weights", policy.weights)
# unsquashed = policy.weights[0] * torch.exp(-0.5 * policy.centers[0, 0]**2 * 1/torch.exp(policy.ln_vars)) + policy.weights[1] * torch.exp(-0.5 * policy.centers[0, 1]**2 * 1/torch.exp(policy.ln_vars))
# print("Unsquashed", unsquashed)
# print("Squashed", policy.squash(unsquashed))
# print(policy(torch.zeros(5, 1)))
