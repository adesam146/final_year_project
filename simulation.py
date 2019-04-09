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
    output: N x T+1
    """
    loc = torch.linspace(start=0, end=end, steps=T+1)
    cov = torch.diag(std_div**2 * torch.ones_like(loc))

    distn = trd.MultivariateNormal(loc, covariance_matrix=cov)

    return distn.sample((N,))

def plot_trajectories(trajectories):
    """
    trajectories: N x T+1
    """
    fig, ax = plt.subplots()

    for t in trajectories:
        t_ = t.numpy()
        ax.scatter(t_, np.zeros_like(t_))

    plt.show()

N = 10
expert_trajectories = get_expert_trajectories(N)
plot_trajectories(expert_trajectories)

from agent import Agent, SimplePolicy
policy = SimplePolicy()
T = 10
agent = Agent(policy, T, dyn_std=0.01)

with torch.no_grad():
    # Only when generating samples from GP posterior do we need the grad wrt policy parameter
    agent.act()

def get_input_output(agent):
    """
    output x: T X D
    output y: T x 1
    """
    x = agent.get_state_action_pairs()
    y = agent.get_curr_trajectory()[1:].reshape(-1, 1)
    return x, y

init_x, init_y = get_input_output(agent)

from forwardmodel import ForwardModel
fm = ForwardModel(init_x, init_y)
fm.train()

# fm.plot_fm_mean(T=T)

init_state_distn = trd.Normal(0, 0.01)

# TODO: LOOP(s) here can perhaps be improved
fm_samples = torch.empty(N, T+1)
for n in range(N):
    x = init_state_distn.sample((1,))
    fm_samples[n, 0] = x
    for t in range(1, T+1):
        x = fm.predict(torch.cat((x, policy.action(x))))
        fm_samples[n, t] = x



