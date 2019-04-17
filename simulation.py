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

GPU = True
device_idx = 0
device = None
if GPU and torch.cuda.is_available():
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")


def get_expert_trajectories(N, end=100, T=10, std_div=0.01):
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


def get_input_output(agent):
    """
    output x: T X D
    output y: T x 1
    """
    x = agent.get_state_action_pairs()
    y = agent.get_curr_trajectory()[1:].reshape(-1, 1)
    return x, y


# This is the number of observed samples from the expert
expert_N = 5
T = 10
expert_trajectories = get_expert_trajectories(expert_N, T=T)

# plot_trajectories(expert_trajectories)

expert_mu = torch.mean(expert_trajectories, dim=0)
expert_var = 1.0/(expert_N-1) * \
    torch.sum((expert_trajectories - expert_mu)**2, dim=0)

expert_distn = trd.MultivariateNormal(expert_mu, torch.diag(expert_var))

policy = SimplePolicy(device)
agent = Agent(policy, T, dyn_std=1e-3, device=device)

with torch.no_grad():
    # Only when generating samples from GP posterior do we need the grad wrt policy parameter
    agent.act()

init_x, init_y = get_input_output(agent)

fm = ForwardModel(init_x.to(device), init_y.to(device), device=device)

# fm.plot_fm_mean(T=T)

init_state_distn = trd.Normal(0, 0.01)

disc = Discrimator(T).to(device)

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
disc_optimizer = torch.optim.Adam(disc.parameters())
policy_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
    policy_optimizer, gamma=0.99)

N = 128

bce_logit_loss = torch.nn.BCEWithLogitsLoss()
real_target = torch.ones(N, 1, device=device)
fake_target = torch.zeros(N, 1, device=device)

num_of_experience = 100
disc_losses = []
policy_losses = []
policies = []
for expr in range(num_of_experience-1):
    fm.learn()

    policy_lr_sch.step()

    for i in range(expr):
        # Optimize policy for given forward model
        # N samples from estimated expert distribution. Shape: N x T+1
        expert_samples = expert_distn.sample((N,)).to(device)

        fm_samples = torch.empty(N, T+1, device=device)
        x = init_state_distn.sample((N,)).to(device)
        fm_samples[:, 0] = x
        for t in range(1, T+1):
            # x = fm.predict(torch.cat((x.view(-1,1), policy.action(x).unsqueeze(0).repeat(N, 1)), dim=1))
            # fm_samples[:, t] = x.view(N)

            x = trd.Normal(x+policy.action(x), scale=1e-3).rsample()
            fm_samples[:,t] = x.view(N)

        # Train Discrimator
        disc_optimizer.zero_grad()

        # We detach forward model samples so that we don't calculate gradients
        # w.r.t the policy parameter here

        loss_fake = bce_logit_loss(disc(fm_samples.detach()), fake_target)
        loss_real = bce_logit_loss(disc(expert_samples), real_target)

        disc_loss = loss_real + loss_fake
        disc_loss.backward()

        disc_losses.append(disc_loss.detach().item())

        disc_optimizer.step()

        # Optimise policy
        policy_optimizer.zero_grad()

        policy_loss = bce_logit_loss(disc(fm_samples), real_target)

        policy_loss.backward()

        policy_losses.append(policy_loss.detach().item())

        policy_optimizer.step()

        print("Experience {}, Iter {}, disc loss: {}, policy loss: {}".format(expr, i, disc_losses[-1], policy_losses[-1]))

    print(policy.theta)
    policies.append(policy.theta.detach().item())
    # Get more experienial data
    with torch.no_grad():
        agent.go_to_beginning()
        agent.act()

    new_x, new_y = get_input_output(agent)

    fm.update_data(new_x, new_y)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

axs[0].plot(disc_losses, label="Losses for discrimator")
axs[1].plot(policy_losses, label="Losses for generator/policy")

for ax in axs:
    ax.set_xlabel("Number of iterations for density matching objective")
    ax.legend()

plt.show()

fig.suptitle(f'Behaviour of density matching objective with {num_of_experience} agent experience with true agent dynamics')
fig.savefig(f'./plots/losses-{num_of_experience}-true-dyanmics.png', format='png')

print(policy.theta)

fig, ax = plt.subplots(figsize=(16,10))

ax.plot(policies)

fig.suptitle("Learnt policy value against number of experience")
fig.savefig(f'./plots/learn_policy_vs_experience_with_true_agent_dynamics.png', format='png')

plt.show()
