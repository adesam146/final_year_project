import torch


class RBFPolicy:
    # Consider making an nn module
    def __init__(self, u_max, input_dim, nbasis, device):
        self.u_max = u_max
        self.input_dim = input_dim
        self.nbasis = nbasis
        self.device = device
        self.weights = torch.randn(nbasis, requires_grad=True, device=device)
        self.centers = torch.randn(
            input_dim, nbasis, requires_grad=True, device=device)
        self.ln_vars = torch.randn(
            input_dim, requires_grad=True, device=device)

    def __call__(self, x):
        """
        x: (batch) x S
        output: batch
        """
        x = x.view(-1, self.input_dim)
        batch = x.shape[0]

        # return torch.matmul(self.weights, torch.exp())
        inv_gamma = torch.diag(1/(torch.exp(self.ln_vars)+1e-12))

        bases = torch.empty(batch, self.nbasis, device=self.device)
        for i in range(self.nbasis):
            x_minus_c_t = x - self.centers[:, i]
            bases[:, i] = torch.diag(torch.chain_matmul(
                x_minus_c_t, inv_gamma, torch.t(x_minus_c_t)))

        return self.u_max * self.squash(torch.matmul(torch.exp(-0.5 * bases), self.weights))

    def squash(self, x):
        """
        Squashing the values in x to be between -1 and 1
        """
        return (9*torch.sin(x) + torch.sin(3*x))/8

    def parameters(self):
        return [self.weights, self.centers, self.ln_vars]

# Testing Policy, TODO: Convert to test
# policy = RBFPolicy(1, input_dim=1, nbasis=2, device=torch.device('cpu'))

# print("Weights", policy.weights)
# unsquashed = policy.weights[0] * torch.exp(-0.5 * policy.centers[0, 0]**2 * 1/torch.exp(policy.ln_vars)) + policy.weights[1] * torch.exp(-0.5 * policy.centers[0, 1]**2 * 1/torch.exp(policy.ln_vars))
# print("Unsquashed", unsquashed)
# print("Squashed", policy.squash(unsquashed))
# print(policy(torch.zeros(5, 1)))
