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
        x: (batch) x D
        output: batch x 1
        """
        batch = 1
        if x.dim() > 1:
            batch = x.shape[0]

        # return torch.matmul(self.weights, torch.exp())
        inv_gamma = torch.diag(1/torch.exp(self.ln_vars))

        bases = torch.empty(batch, self.nbasis, device=self.device)
        for i in range(self.nbasis):
            x_minus_c_t = x.view(batch, self.input_dim) - self.centers[:, i]
            bases[:, i] = torch.diag(torch.chain_matmul(
                x_minus_c_t, inv_gamma, torch.t(x_minus_c_t)))

        # Wrapping the result in a 1D tensor of size 1
        return self.u_max * self.squash(torch.matmul(torch.exp(-0.5 * bases), self.weights))

    def squash(self, x):
        """
        Squashing the values in x to be between -1 and 1
        """
        return (9*torch.sin(x) + torch.sin(3*x))/8

    def parameters(self):
        return [self.weights, self.centers, self.ln_vars]
