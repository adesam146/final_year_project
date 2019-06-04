import numpy as np
import torch
import torch.distributions as trd
from cartpole.utils import get_training_data, get_expert_data, convert_to_aux_state
import matplotlib.pyplot as plt


# TODO: Consider
# gpytorch.settings.detach_test_caches(state=True)
# https://gpytorch.readthedocs.io/en/latest/settings.html?highlight=fantasy

# Set random seed to ensure that your results are reproducible.
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

GPU = False
device_idx = 0
device = None
if GPU and torch.cuda.is_available():
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")

expert_samples = get_expert_data().to(device)
N = expert_samples.shape[0]
T = expert_samples.shape[1]
state_dim = expert_samples.shape[2]

# means are T x state_dim

def estimate_mean_and_covariance():
  # means are T x state_dim
  means = torch.mean(expert_samples, dim=0)

  covars = torch.zeros(T, state_dim, state_dim, device=device)

  for n in range(N):
    for t in range(T):
      x_minus_u = expert_samples[n, t].view(-1, 1) - means[t].view(-1, 1)
      covars[t] += torch.matmul(x_minus_u, torch.t(x_minus_u))

  covars *= 1/(N-1)

estimate_mean_and_covariance()



