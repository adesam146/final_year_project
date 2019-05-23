import numpy as np
import torch
import torch.distributions as trd
import torch.nn.functional as F
from cartpole.agent import CartPoleAgent
from cartpole.utils import get_training_data, get_expert_data, convert_to_aux_state, plot_trajectories, plot_gp_trajectories
from forwardmodel import ForwardModel
from rbf_policy import RBFPolicy
from discrimator import Discrimator
import matplotlib.pyplot as plt
import gpytorch
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", help="Train using a GPU if available", action="store_true")
parser.add_argument("--result_dir_name", help="Name of directory to place results")
parser.add_argument("--policy_lr", type=float, help="Learning rate for the policy, default is 1e-2")
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
result_dir_name = f'{args.result_dir_name or "result"}'
result_dir = os.path.join(script_dir, result_dir_name)
count = 1
while os.path.isdir(result_dir):
    result_dir = os.path.join(script_dir, result_dir_name+f'{count}')
    count += 1
os.makedirs(result_dir)
plot_dir = os.path.join(result_dir, 'plots/')
os.makedirs(plot_dir)
variables_file = os.path.join(result_dir, 'variables.json')

# TODO: Consider
# gpytorch.settings.detach_test_caches(state=True)
# https://gpytorch.readthedocs.io/en/latest/settings.html?highlight=fantasy

# Set random seed to ensure that your results are reproducible.
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

GPU = args.gpu and torch.cuda.is_available()
device_idx = 0
device = None
if GPU:
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")

expert_samples = get_expert_data().to(device)
T = expert_samples.shape[1]
# This is the number of samples from each source (expert/predicted) to be compared against each other
N = expert_samples.shape[0]

state_dim = 4
aux_state_dim = 5
action_dim = 1
dt = 0.1
time = dt * T

# *** POLICY SETUP ***
policy = RBFPolicy(u_max=10, input_dim=aux_state_dim, nbasis=10, device=device)
policy_lr = args.policy_lr or 1e-2

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
N_x0 = 10

# *** DISCRIMATOR SETUP ***
disc = Discrimator(T, D=state_dim).to(device)
disc_optimizer = torch.optim.Adam(disc.parameters())

bce_logit_loss = torch.nn.BCEWithLogitsLoss()
real_target = torch.ones(N, 1, device=device)
fake_target = torch.zeros(N*N_x0, 1, device=device)
real_target_for_policy = torch.ones(N*N_x0, 1, device=device)

num_of_experience = 50

# Write to a json file all defined variables before training starts
with open(variables_file, 'w') as fp:
    json.dump(locals(), fp, skipkeys=True, sort_keys=True, default=lambda obj: "The object can't be json serialized")

with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
    for expr in range(1, num_of_experience+1):
        print("Experience:", expr)
        fm.learn()

        policy_lr_sch.step()

        for i in range(50):
            # Optimize policy for given forward model

            def get_samples_and_log_prob(N, N_x0=10):
                fm_samples = expert_samples.new_empty(N_x0 * N, T, state_dim)
                actions = expert_samples.new_empty(N_x0 * N, T, action_dim)
                log_prob = fm_samples.new_zeros(N_x0 * N)
                x0s = init_state_distn.sample((N_x0,)).to(device)
                for j, x0 in enumerate(x0s):
                    for i in range(N):
                        x = x0
                        for t in range(T):
                            aux_x = convert_to_aux_state(x, state_dim)
                            x_t_u_t = torch.cat(
                                (aux_x, policy(aux_x).view(-1, action_dim)), dim=1)
                            y, log_prob_t = fm.predict(x_t_u_t)
                            log_prob[i + N*j] += log_prob_t

                            fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
                            x = x + y.view(state_dim)

                            fm_samples[i + N*j, t] = x
                            actions[i + N*j, t] = x_t_u_t[:, -
                                                    action_dim:].detach().squeeze()
                        fm.clear_fantasy_data()

                return fm_samples, log_prob, actions, x0s

            # Train Discrimator
            disc.enable_parameters_grad()
            disc_optimizer.zero_grad()
            fm_samples, log_prob, _, _ = get_samples_and_log_prob(N, N_x0)

            # We detach forward model samples so that we don't calculate gradients w.r.t the policy parameter here
            disc_loss = bce_logit_loss(disc(fm_samples.detach(
            )), fake_target) + bce_logit_loss(disc(expert_samples), real_target)

            disc_loss.backward()
            disc_optimizer.step()

            # To avoid having to calculate gradients of discrimator
            disc.enable_parameters_grad(enable=False)

            # Optimise policy
            policy_optimizer.zero_grad()
            policy_loss = torch.mean(log_prob.view(-1, 1) * F.binary_cross_entropy_with_logits(
                disc(fm_samples), real_target_for_policy, reduction='none') - log_prob.view(-1, 1))
            policy_loss.backward()
            policy_optimizer.step()

            print(
                f"Experience {expr}, Iter {i}, disc loss: {disc_loss.detach().item()}, policy loss: {policy_loss.detach()}, policy weights grad: {policy.weights.grad}, policy ln_vars grad: {policy.ln_vars.grad}, policy centers grad: {policy.centers.grad}")

        # Get more experienial data
        with torch.no_grad():
            s_a_pairs, traj = agent.act()
            new_x, new_y = get_training_data(s_a_pairs, traj)

        fm.update_data(new_x, new_y)

        # Plotting progress

        # Plotting expert trajectory
        fig, axs = plot_trajectories(
            expert_samples, T=T, color='blue', with_x0=False)

        # Plotting result of rollout with current policy
        for roll in range(10):
            with torch.no_grad():
                s_a_pairs, traj = agent.act()
            # Plot states
            for i, ax in enumerate(np.ravel(axs[:state_dim])):
                ax.plot(np.arange(0, T+1), traj.cpu().numpy()
                        [:, i], color='red', alpha=0.2)
            # Plot action
            axs[-1].plot(np.arange(0, T), s_a_pairs[:, -
                                                    1].cpu().numpy(), color='red', alpha=0.2)
        fig.suptitle(
            f"State values of expert vs learner with {expr} number of experience")

        fig.tight_layout()

        fig.savefig(
            plot_dir + f'expert_vs_learner_{expr}.png', format='png')

        # Plotting prediction of GP under current policy
        with torch.no_grad():
            samples, _, actions, x0s = get_samples_and_log_prob(N=10, N_x0=N_x0)
        plot_gp_trajectories(torch.cat((x0s.repeat_interleave(10, dim=0).unsqueeze(
            1), samples), dim=1), actions, T=T, plot_dir=plot_dir, expr=expr)
