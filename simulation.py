from forwardmodel import ForwardModel
from agent import Agent, SimplePolicy
import numpy as np
import torch
import torch.distributions as trd
from matplotlib import pyplot as plt
from discrimator import Discrimator

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


# This is the number of observed samples from the expert
expert_N = 5
expert_trajectories = get_expert_trajectories(expert_N)

# plot_trajectories(expert_trajectories)

expert_mu = torch.mean(expert_trajectories, dim=0)
expert_var = 1.0/(expert_N-1) * \
    torch.sum((expert_trajectories - expert_mu)**2, dim=0)
expert_distn = trd.MultivariateNormal(expert_mu, torch.diag(expert_var))

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

fm = ForwardModel(init_x, init_y)
fm.learn()

delete = fm.predict(torch.tensor([2.0, 2]))

# fm.plot_fm_mean(T=T)

init_state_distn = trd.Normal(0, 0.01)

disc = Discrimator(T)

policy_optimizer = torch.optim.Adam(policy.parameters())
disc_optimizer = torch.optim.Adam(disc.parameters())

N = 10
real_target = torch.ones(N, 1)
fake_target = torch.zeros(N, 1)

bce_logit_loss = torch.nn.BCEWithLogitsLoss()

for i in range(50):
    print(i)

    # TODO: LOOP(s) here can perhaps be improved
    
    # N samples from estimated expert distribution. Shape: N x T+1
    expert_samples = expert_distn.sample((N,))

    fm_samples = torch.empty(N, T+1)
    for n in range(N):
        x = init_state_distn.sample((1,))
        fm_samples[n, 0] = x
        for t in range(1, T+1):
            x = fm.predict(torch.cat((x, policy.action(x))))
            fm_samples[n, t] = x

    # Train Discrimator
    disc_optimizer.zero_grad()

    # We detach forward model samples so that we don't calculate gradients
    # w.r.t the policy parameter here
    loss_fake = bce_logit_loss(disc(fm_samples.detach()), fake_target)
    loss_real = bce_logit_loss(disc(expert_samples), real_target)

    disc_loss = loss_real + loss_fake
    disc_loss.backward()

    disc_optimizer.step()

    # Optimise policy
    policy_optimizer.zero_grad()

    policy_loss = bce_logit_loss(disc(fm_samples), real_target)

    policy_loss.backward()

    policy_optimizer.step()

    print(policy.theta)
