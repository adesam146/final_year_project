import json
import os
from datetime import datetime

import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as trd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cartpole.agent import CartPoleAgent
from cartpole.args import args
from cartpole.cartpole_setup import CartPoleSetup
from cartpole.optimal_gp import OptimalGP
from cartpole.optimal_policy import OptimalPolicy
from cartpole.pathwise_grad import pathwise_grad
from cartpole.score_function import score_function_training
from cartpole.state_to_state import state_to_state
from cartpole.utils import (convert_to_aux_state, get_expert_data,
                            get_training_data, plot_gp_trajectories,
                            plot_progress, plot_trajectories,
                            sample_trajectories, save_current_state)
from discriminator import ConvDiscriminator, Discriminator
from forwardmodel import ForwardModel
from nn_policy import DeepNNPolicy, NNPolicy
from rbf_policy import RBFPolicy
from ss_discriminator import SSDiscriminator

font = {'size': 14}
matplotlib.rc('font', **font)
matplotlib.use('Agg')

# *** RESULTS LOGGING SETUP ***
script_dir = os.path.dirname(__file__)
exec_datetime = datetime.now().strftime('%Y-%m-%d-T-%H-%M-%S')
result_dir_name = args.result_dir_name or f"result-{exec_datetime}"
result_dir = os.path.join(os.path.join(script_dir, "results"), result_dir_name)

# In case the provided name already exist
if os.path.isdir(result_dir):
    result_dir = result_dir + f'-{exec_datetime}'

os.makedirs(result_dir)

plot_dir = os.path.join(result_dir, 'plots/')
os.makedirs(plot_dir)
variables_file = os.path.join(result_dir, 'variables.json')
description_file = os.path.join(result_dir, 'description.txt')
with open(description_file, 'w') as fp:
    fp.write(f'{args.description}')

# Set random seed to ensure that results are reproducible.
fix_seed = args.fix_seed or False
if fix_seed:
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

# *** POLICY OPTIMISATION ALGORITHM CHOICE ***
use_score_func_grad = args.use_score_func_grad
use_pathwise_grad = args.use_pathwise_grad
use_max_log_prob = args.use_max_log_prob
use_state_to_state = args.use_state_to_state

# *** GET EXPERT SAMPLES ***
expert_samples = get_expert_data()

# N is the number of samples from each source (expert/predicted) to be compared against each other
setup = CartPoleSetup(
    N=expert_samples.shape[0], T=args.T or expert_samples.shape[1]-1)

# Determine whether expert x0 is needed or not
expert_sample_start = 0
if use_score_func_grad or args.with_x0:
    expert_sample_start = 1
# Restrict samples to specfied horizon T
expert_dl = DataLoader(TensorDataset(
    expert_samples[:, expert_sample_start:setup.T+1, :]), batch_size=args.batch_size or setup.N)

# *** POLICY SETUP ***
policy_dir = os.path.join(result_dir, 'policy/')
os.makedirs(policy_dir)
if args.policy == 'rbf':
    policy = RBFPolicy(u_max=10, input_dim=setup.aux_state_dim,
                       nbasis=50, device=device)
elif args.policy == 'nn':
    policy = NNPolicy(u_max=10, input_dim=setup.aux_state_dim).to(device)
elif args.policy == 'optimal':
    policy = OptimalPolicy(u_max=10, device=device)
else:
    policy = DeepNNPolicy(u_max=10, input_dim=setup.aux_state_dim).to(device)
policy_lr = args.policy_lr or 1e-2

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
# policy_optimizer = torch.optim.LBFGS(policy.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
policy_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
    policy_optimizer, gamma=(0.9)**(1/100))

# *** AGENT SETUP ***
agent = CartPoleAgent(dt=setup.dt, T=setup.T, policy=policy,
                      measurement_var=torch.diag(0.01**2 * torch.ones(setup.state_dim)), device=device)

# *** FIRST RANDOM ROLLOUT ***
with torch.no_grad():
    s_a_pairs, traj = agent.act()
    fig, axs = plot_trajectories(samples=traj.unsqueeze(
        0), actions=s_a_pairs[:, -setup.action_dim:].unsqueeze(0), T=setup.T)
    fig.tight_layout()
    fig.savefig(plot_dir + 'initial_rollout.pdf', format='pdf')
    plt.close(fig)
    init_x, init_y = get_training_data(s_a_pairs, traj)

# *** FORWARD MODEL SETUP ***
fm_dir = os.path.join(result_dir, 'fm/')
os.makedirs(fm_dir)
fm = ForwardModel(init_x=init_x.to(device),
                  init_y=init_y.to(device), D=setup.state_dim, S=setup.aux_state_dim, F=setup.action_dim, device=device, save_dir=fm_dir)
# fm = OptimalGP(device=device, save_dir=fm_dir)


# *** INITIAL STATE DISTRIBUTION FOR GP PREDICTION ***
init_state_distn = trd.MultivariateNormal(loc=torch.zeros(
    setup.state_dim, device=device), covariance_matrix=torch.diag(0.1**2 * torch.ones(setup.state_dim, device=device)))
N_x0 = 10

# *** DISCRIMINATOR SETUP ***
use_conv_disc = args.use_conv_disc
disc_dir = os.path.join(result_dir, 'disc/')
os.makedirs(disc_dir)

if use_state_to_state:
    disc = SSDiscriminator(D=setup.state_dim)
elif use_conv_disc:
    disc = ConvDiscriminator(T=setup.T, D=setup.state_dim,
                             with_x0=expert_sample_start == 0).to(device)
else:
    disc = Discriminator(T=setup.T, D=setup.state_dim).to(device)

disc_lr = args.disc_lr or 1e-2
disc_optimizer = torch.optim.Adam(disc.parameters())
# disc_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
#     disc_optimizer, gamma=(0.9)**(1/100))

num_of_experience = args.num_expr or 50
policy_iter = args.policy_iter or 50

# Write to a json file all defined variables before training starts
with open(variables_file, 'w') as fp:
    json.dump({**locals(), **setup.__dict__}, fp, skipkeys=True, sort_keys=True,
              default=lambda obj: type(obj).__name__, indent=2)

with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
    for expr in range(1, num_of_experience+1):
        print("Experience:", expr)
        fm.learn()

        print("Policy Optimizer learning rate:")
        for param_group in policy_optimizer.param_groups:
            print(param_group['lr'])

        for i in range(policy_iter):
            avg_policy_loss = 0
            avg_disc_loss = 0
            for batch, (expert_samples,) in enumerate(expert_dl):
                expert_samples = expert_samples.to(device)
                if use_score_func_grad:
                    disc_loss, policy_loss, samples, actions = score_function_training(
                        setup, N_x0, expert_samples, policy, fm, disc, disc_optimizer, policy_optimizer, init_state_distn)

                    avg_policy_loss += policy_loss.detach()
                    avg_disc_loss += disc_loss.detach()

                if use_pathwise_grad:
                    disc_loss, policy_loss, samples, actions = pathwise_grad(
                        setup, expert_samples, policy, fm, disc, disc_optimizer, policy_optimizer, init_state_distn)

                    avg_policy_loss += policy_loss.detach()
                    avg_disc_loss += disc_loss.detach()

                if use_state_to_state:
                    disc_loss, policy_loss, samples, actions = state_to_state(
                        setup, expert_samples, policy, fm, disc, disc_optimizer, policy_optimizer, init_state_distn)

                    avg_policy_loss += policy_loss.detach()
                    avg_disc_loss += disc_loss.detach()

                if use_max_log_prob:
                    policy_optimizer.zero_grad()
                    policy_loss = 0
                    for n in range(setup.N):
                        for t in range(setup.T):
                            # This can perhaps also be done in a batch form
                            aux_x = convert_to_aux_state(
                                expert_samples[n, t], setup.state_dim)
                            dyn_model = fm.predictive_distn(
                                torch.cat((aux_x, policy(aux_x).view(-1, setup.action_dim)), dim=1))
                            policy_loss += -dyn_model.log_prob(
                                (expert_samples[n, t+1] - expert_samples[n, t]).view(setup.state_dim, -1)).sum()

                            del dyn_model

                    policy_loss *= 1.0/setup.N
                    policy_loss.backward()
                    policy_optimizer.step()

                    avg_policy_loss += policy_loss.detach()

            # Plot and print out progress
            avg_policy_loss *= 1/len(expert_dl)
            avg_disc_loss *= 1/len(expert_dl)
            if not use_max_log_prob:
                if i % 10 == 0:
                    plot_gp_trajectories(
                        samples.detach(), actions.detach(), T=setup.T, plot_dir=plot_dir, title=f'Predicted trajectories with {expr} interactions and at {i} policy iterations (before the {i+1}th parameter update).', file_name=f'training_trajs_expr-{expr}-policy_iter-{i}')

                del samples, actions

                print(
                    f"Experience {expr}, Iter {i}, disc loss: {avg_disc_loss.detach().item()}, policy loss: {avg_policy_loss.detach()}")
            else:
                print(
                    f"Experience {expr}, Iter {i}, policy loss: {avg_policy_loss.detach().item()}")

        policy_lr_sch.step()

        policy.eval()

        # Plotting progress
        plot_progress(setup, expr, agent, plot_dir, policy,
                      init_state_distn, fm, expert_samples, N_x0)

        # Saving Current state
        save_current_state(expr, fm, disc, disc_optimizer, disc_dir,
                           policy, policy_optimizer, policy_lr_sch, policy_dir)

        # Get more experienial data
        with torch.no_grad():
            s_a_pairs, traj = agent.act()
            new_x, new_y = get_training_data(s_a_pairs, traj)

        fm.update_data(new_x, new_y)

        policy.train()
