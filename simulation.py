import numpy as np
import torch
import torch.distributions as trd
from matplotlib import pyplot as plt

# Set random seed to ensure that your results are reproducible.
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


def get_expert_trajectories(N, end=100, T=10, std_div=1):
    """
    """
    loc = torch.linspace(start=0, end=end, steps=T)
    cov = torch.diag(std_div**2 * torch.ones_like(loc))

    distn = trd.MultivariateNormal(loc, covariance_matrix=cov)

    return distn.sample((N,)).view(-1, 1)


fig, ax = plt.subplots()

# trajectories is a N x 1 tensor
trajectories = get_expert_trajectories(10)
for t in trajectories:
    t_ = t.numpy()
    ax.scatter(t_, np.zeros_like(t_))

plt.show()
