from rbf_policy import RBFPolicy
from forwardmodel import ForwardModel
from agent import Agent, SimplePolicy
import numpy as np
import torch
import torch.distributions as trd
import torch.nn.functional as F
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


def get_expert_trajectories(N, end=10, T=10, std_div=0.01):
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
    output x: T X D+F
    output y: T x 1
    """
    x = agent.get_state_action_pairs()
    y = agent.get_curr_trajectory()[1:].reshape(-1, 1) - x[:, 0].view(-1, 1)
    return x, y


state_dim = 1
action_dim = 1

# This is the number of observed samples from the expert
expert_N = 5
T = 10

expert_trajectories = get_expert_trajectories(expert_N, T=T)
print(expert_trajectories[0])

# plot_trajectories(expert_trajectories)

expert_mu = torch.mean(expert_trajectories, dim=0)
expert_var = 1.0/(expert_N-1) * \
    torch.sum((expert_trajectories - expert_mu)**2, dim=0)

expert_distn = trd.MultivariateNormal(expert_mu, torch.diag(expert_var))

# policy = SimplePolicy(device)
policy = RBFPolicy(u_max=20, input_dim=state_dim, nbasis=5, device=device)
dyn_std = 1e-2
agent = Agent(policy, T, dyn_std=dyn_std, device=device)

with torch.no_grad():
    # Only when generating samples from GP posterior do we need the grad wrt policy parameter
    agent.act()

init_x, init_y = get_input_output(agent)

fm = ForwardModel(init_x=init_x.to(device),
                  init_y=init_y.to(device), D=state_dim, S=state_dim, F=action_dim, device=device)

# fm.plot_fm_mean(T=T)

# We are free to choose what the initial state distribution of the agent is so setting it to same as expert.
init_state_distn = trd.MultivariateNormal(torch.tensor(
    [expert_mu[0]]), torch.diag(torch.sqrt(expert_var[0]).view(state_dim)))

disc = Discrimator(T).to(device)

policy_lr = 1e-2
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
disc_optimizer = torch.optim.Adam(disc.parameters())
policy_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
    policy_optimizer, gamma=0.99)

# This is the number of samples from each distribution to be compared
# against each other
N = 256

bce_logit_loss = torch.nn.BCEWithLogitsLoss()
real_target = torch.ones(N, 1, device=device)
fake_target = torch.zeros(N, 1, device=device)

num_of_experience = 50
disc_losses = []
policy_losses = []
policies = []
for expr in range(1, num_of_experience+1):
    fm.learn()

    policy_lr_sch.step()

    for i in range(expr):
        # Optimize policy for given forward model
        # N samples from estimated expert distribution. Shape: N x T
        expert_samples = expert_distn.sample((N,))[:, 1:].to(device)

        fm_samples = torch.empty(N, T, device=device)
        x0s = init_state_distn.sample((N,)).to(device)
        log_prob = fm_samples.new_zeros(N)

        for j, x in enumerate(x0s):
            for t in range(T):
                x_t_u_t = torch.cat(
                    (x.view(-1, 1), policy(x).view(-1, 1)), dim=1)
                y, log_prob_t = fm.predict(x_t_u_t)
                log_prob[j] += log_prob_t
                fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
                x = x + y.view(state_dim)
                fm_samples[j, t] = x
            fm.clear_fantasy_data()

        # Train Discrimator
        disc_optimizer.zero_grad()

        # We detach forward model samples so that we don't calculate gradients
        # w.r.t the policy parameter here

        loss_fake = bce_logit_loss(disc(fm_samples.detach().unsqueeze(2)), fake_target)
        loss_real = bce_logit_loss(disc(expert_samples.unsqueeze(2)), real_target)

        disc_loss = loss_real + loss_fake
        disc_loss.backward()
        disc_optimizer.step()
        disc_losses.append(disc_loss.detach().item())

        # Optimise policy
        policy_optimizer.zero_grad()
        policy_loss = torch.mean(log_prob.view(-1, 1) * F.binary_cross_entropy_with_logits(
            disc(fm_samples.detach().unsqueeze(2)), real_target, reduction='none') - log_prob.view(-1, 1))
        policy_loss.backward()
        policy_optimizer.step()
        policy_losses.append(policy_loss.detach().item())

        print("Experience {}, Iter {}, disc loss: {}, policy loss: {}".format(
            expr, i, disc_losses[-1], policy_losses[-1]))

    # print(policy.theta)
    # policies.append(policy.theta.detach().item())

    # Get more experienial data
    with torch.no_grad():
        agent.go_to_beginning()
        agent.act()

    new_x, new_y = get_input_output(agent)

    fm.update_data(new_x, new_y)

    fig, ax = plt.subplots()

    ax.plot(np.arange(0, T+1), agent.get_curr_trajectory().cpu().numpy())

    ax.set_xlabel("Time steps")
    ax.set_ylabel(r'$x_t$')
    ax.set_title(
        r"$x_t$ learner with {} number of training experience".format(expr))

    fig.savefig(
        f'./plots/learner_{expr}-T-{T}-N-{N}.png', format='png')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

axs[0].plot(disc_losses, label="Losses for discrimator")
axs[1].plot(policy_losses, label="Losses for generator/policy")

for ax in axs:
    ax.set_xlabel("Number of iterations for density matching objective")
    ax.legend()

fig.suptitle(
    f'Behaviour of density matching objective with {num_of_experience} agent experience with true agent dynamics')
fig.savefig(
    f'./plots/compareT/losses-{num_of_experience}-T-{T}-lr-{policy_lr}-dynstd-{dyn_std}-N-{N}.png', format='png')

# print(policy.theta)

# fig, ax = plt.subplots(figsize=(16, 10))

# ax.plot(policies)

# fig.suptitle("Learnt policy value against number of experience")
# fig.savefig(
#     f'./plots/compareT/learn_policy_vs_experience-{num_of_experience}-T-{T}-lr-{policy_lr}-dynstd-{dyn_std}-N-{N}.png', format='png')

# plt.show()
