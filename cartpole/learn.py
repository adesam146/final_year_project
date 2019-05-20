import numpy as np
import torch
import torch.distributions as trd
import torch.nn.functional as F
from cartpole.agent import CartPoleAgent
from cartpole.utils import get_training_data, get_expert_data, convert_to_aux_state
from forwardmodel import ForwardModel
from rbf_policy import RBFPolicy
from discrimator import Discrimator
import matplotlib.pyplot as plt
import gpytorch


# TODO: Consider
# gpytorch.settings.detach_test_caches(state=True)
# https://gpytorch.readthedocs.io/en/latest/settings.html?highlight=fantasy

# Set random seed to ensure that your results are reproducible.
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

GPU = True
device_idx = 0
device = None
if GPU and torch.cuda.is_available():
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")

expert_samples = get_expert_data().to(device)

state_dim = 4
aux_state_dim = 5
action_dim = 1
dt = 0.1
time = 4.0
T = int(np.ceil(time/dt))
# This is the number of samples from each source (expert/predicted) to be compared against each other
N = expert_samples.shape[0]

# *** POLICY SETUP ***
policy = RBFPolicy(u_max=10, input_dim=aux_state_dim, nbasis=10, device=device)
policy_lr = 1e-2

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
# policy_optimizer = torch.optim.LBFGS(policy.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
policy_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
    policy_optimizer, gamma=(0.9)**(1/100))

# *** AGENT SETUP ***
measurement_var = torch.diag(0.01**2 * torch.ones(state_dim))
agent = CartPoleAgent(dt=dt, time=time, policy=policy,
                      measurement_var=measurement_var, device=device)

# *** FIRST RANDOM ROLLOUT ***
with torch.no_grad():
    s_a_pairs, traj = agent.act()
    init_x, init_y = get_training_data(s_a_pairs, traj)

# *** FORWARD MODEL SETUP ***
fm = ForwardModel(init_x=init_x.to(device),
                  init_y=init_y.to(device), D=state_dim, S=aux_state_dim, F=action_dim, device=device)

# *** INITIAL STATE DISTRIBUTION FOR GP PREDICTION ***
init_state_distn = trd.MultivariateNormal(loc=torch.zeros(
    state_dim), covariance_matrix=torch.diag(0.1**2 * torch.ones(state_dim)))

# *** DISCRIMATOR SETUP ***
disc = Discrimator(T, D=state_dim).to(device)
disc_optimizer = torch.optim.Adam(disc.parameters())


bce_logit_loss = torch.nn.BCEWithLogitsLoss()
real_target = torch.ones(N, 1, device=device)
fake_target = torch.zeros(N, 1, device=device)

num_of_experience = 50

with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
    for expr in range(1, num_of_experience+1):
        print("Experience:", expr)
        fm.learn()

        policy_lr_sch.step()

        for i in range(expr):
            # Optimize policy for given forward model

            def get_samples_and_log_prob():
                fm_samples = expert_samples.new_empty(N, T, state_dim)
                log_prob = fm_samples.new_zeros(N)
                x0s = init_state_distn.sample((N,)).to(device)
                for j, x in enumerate(x0s):
                    for t in range(T):
                        aux_x = convert_to_aux_state(x, state_dim)
                        x_t_u_t = torch.cat(
                            (aux_x, policy(aux_x).view(-1, action_dim)), dim=1)
                        y, log_prob_t = fm.predict(x_t_u_t)
                        log_prob[j] += log_prob_t
                        fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
                        x = x + y.view(state_dim)
                        fm_samples[j, t] = x
                    fm.clear_fantasy_data()

                return fm_samples, log_prob

            # Train Discrimator
            disc_optimizer.zero_grad()
            fm_samples, log_prob = get_samples_and_log_prob()
            # We detach forward model samples so that we don't calculate gradients w.r.t the policy parameter here
            disc_loss = bce_logit_loss(disc(fm_samples.detach()), fake_target) + bce_logit_loss(disc(expert_samples), real_target)

            disc_loss.backward()
            disc_optimizer.step()

            # Optimise policy
            policy_optimizer.zero_grad()
            policy_loss = torch.mean(log_prob.view(-1, 1) * F.binary_cross_entropy_with_logits(
                disc(fm_samples), real_target, reduction='none') - log_prob.view(-1, 1))
            policy_loss.backward()
            policy_optimizer.step()

            print("Experience {}, Iter {}, disc loss: {}, policy loss: {}".format(
                expr, i, disc_loss.detach().item(), policy_loss))

        # Get more experienial data
        with torch.no_grad():
            s_a_pairs, traj = agent.act()
            new_x, new_y = get_training_data(s_a_pairs, traj)

        fm.update_data(new_x, new_y)

        # Plotting progress
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        state_names = [r'$x$', r'$v$', r'$\dot{\theta}$', r'$\theta$']
        for i, ax in enumerate(np.ravel(axs)):
            # Plotting expert theta trajectory
            for n in range(N):
                ax.plot(np.arange(1, T+1), expert_samples.cpu().numpy()
                        [n, :, i], alpha=0.15, color='blue')
            ax.set_xlabel("Time steps")
            ax.set_ylabel(state_names[i])

        for roll in range(10):
            with torch.no_grad():
                _, traj = agent.act()
            for i, ax in enumerate(np.ravel(axs)):
                ax.plot(np.arange(0, T+1), traj.cpu().numpy()
                        [:, i], color='red', alpha=0.2)

        fig.suptitle(
            "State values of expert vs learner with {} number of experience".format(expr))

        fig.savefig(
            f'./cartpole/plots/0.01lr/expert_vs_learner_{expr}_2.png', format='png')
