import numpy as np
import torch
import torch.distributions as trd
from cartpole.agent import CartPoleAgent
from cartpole.utils import get_train_y, get_expert_data
from forwardmodel import ForwardModel
from rbf_policy import RBFPolicy
from discrimator import Discrimator


# TODO: Consider
# gpytorch.settings.detach_test_caches(state=True)
# https://gpytorch.readthedocs.io/en/latest/settings.html?highlight=fantasy

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

expert_samples = get_expert_data().to(device)

state_dim = 4
action_dim = 1
policy = RBFPolicy(u_max=10, input_dim=state_dim, nbasis=10, device=device)
dt = 0.1
time = 4
T = int(np.ceil(4/dt))
dyn_var = torch.diag(0.01**2 * torch.ones(state_dim))
agent = CartPoleAgent(dt=dt, time=time, policy=policy,
                      dyn_var=dyn_var, device=device)

with torch.no_grad():
    # Only when generating samples from GP posterior do we need the grad wrt policy parameter
    s_a_pairs, traj = agent.act()
    init_y = get_train_y(traj)

fm = ForwardModel(init_x=s_a_pairs.to(device),
                  init_y=init_y.to(device), device=device)

init_state_distn = trd.MultivariateNormal(loc=torch.zeros(
    state_dim), covariance_matrix=torch.diag(0.1**2 * torch.ones(state_dim)))

disc = Discrimator(T, D=state_dim).to(device)

policy_lr = 0.5
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
disc_optimizer = torch.optim.Adam(disc.parameters())
policy_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
    policy_optimizer, gamma=0.99)

# This is the number of samples from each source (expert/predicted) to be compared
# against each other
N = expert_samples.shape[0]

bce_logit_loss = torch.nn.BCEWithLogitsLoss()
real_target = torch.ones(N, 1, device=device)
fake_target = torch.zeros(N, 1, device=device)

num_of_experience = 50
disc_losses = []
policy_losses = []

for expr in range(num_of_experience-1):
    fm.learn()

    policy_lr_sch.step()

    for i in range(expr):
        # Optimize policy for given forward model

        fm_samples = torch.empty(N, T, state_dim, device=device)
        x = init_state_distn.sample((N,)).to(device)
        for t in range(T):
            y = fm.predict(
                torch.cat((x, policy(x).view(-1, action_dim)), dim=1)
            )
            x = x + y
            fm_samples[:, t] = x

        # Train Discrimator
        disc_optimizer.zero_grad()

        # We detach forward model samples so that we don't calculate gradients
        # w.r.t the policy parameter here
        loss_fake = bce_logit_loss(disc(fm_samples.detach()), fake_target)
        loss_real = bce_logit_loss(disc(expert_samples), real_target)

        disc_loss = loss_real + loss_fake
        disc_loss.backward()
        disc_optimizer.step()
        disc_losses.append(disc_loss.detach().item())

        # Optimise policy
        policy_optimizer.zero_grad()
        policy_loss = bce_logit_loss(disc(fm_samples), real_target)
        policy_loss.backward()
        policy_optimizer.step()
        policy_losses.append(policy_loss.detach().item())

        print("Experience {}, Iter {}, disc loss: {}, policy loss: {}".format(
            expr, i, disc_losses[-1], policy_losses[-1]))

    # Get more experienial data
    with torch.no_grad():
        new_x, traj = agent.act()
        new_y = get_train_y(traj)

    fm.update_data(new_x, new_y)
