import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def get_expert_data(N=64):
    """
    output: N X T+1 x 4
    """
    result = torch.empty(N, 41, 4)

    for i in range(N):
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/expert_data/clockwise/data{i+1}.tsv', delimiter='\t')

        result[i, :, :] = torch.from_numpy(
            df.loc[:, ['x', 'v', 'dtheta', 'theta']].to_numpy())

    return result


def get_training_data(s_a_pairs, traj):
    """
    s_a_pairs: T x D+F
    traj: T+1 x D
    output x: T x D+1+F
    output y: T x D
    Converts a trajectory into the form to be used a target data for the GP,
    i.e into x_t+1 - x_t
    """
    D = traj.shape[-1]

    # replaces theta in the state with sin(theta) and cos(theta) to exploit
    # the wrapping around of angles
    x = torch.cat((convert_to_aux_state(
        s_a_pairs[:, :D], D), s_a_pairs[:, D:]), dim=1)

    y = traj[1:] - traj[:-1]
    return x, y + 0.01 * torch.randn_like(y)


def convert_to_aux_state(state, D):
    """
    state: (N x) D
    output: N x S
    """
    state = state.view(-1, D)

    return torch.cat((state[:, :D-1], torch.sin(state[:, D-1:D]), torch.cos(state[:, D-1:D])), dim=1)


def plot_trajectories(samples, T, actions=None, color=None, alpha=0.15, with_x0=True):
    """
    Creates a figure and axes for the states and actions.
    samples: N x T(+1) x D=4
    actions: N x T, if None then empty plot added
    output: fig and an array of axis for each state dimension D.
    """
    assert samples.shape[2] == 4

    if with_x0:
        assert samples.shape[1] == T+1
        x = np.arange(0, T+1)
    else:
        assert samples.shape[1] == T
        x = np.arange(1, T+1)

    fig = plt.figure(figsize=(16, 10))
    axs = []

    state_action_names = [r'$x$', r'$v$',
                          r'$\dot{\theta}$', r'$\theta$', r'$u$']

    nrows = 3
    ncols = 2

    # Plot states
    for i in range(samples.shape[2]):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.plot(x, torch.t(samples[:, :, i]).cpu(
        ).numpy(), alpha=alpha, color=color)

        axs.append(ax)

    # Plot actions
    ax = fig.add_subplot(nrows, ncols, 5)
    if actions is not None:
        ax.plot(np.arange(0, T), torch.t(actions[:, :, 0]).cpu(
        ).numpy(), alpha=alpha, color=color)
    axs.append(ax)

    for i, ax in enumerate(axs):
        ax.set_xlabel("Time steps")
        ax.set_ylabel(state_action_names[i])

    return fig, axs


def plot_gp_trajectories(fm_samples, actions, T, plot_dir, expr):
    gp_fig, _ = plot_trajectories(
        fm_samples, actions=actions, T=T, color='green')
    gp_fig.suptitle(
        f"Predicted state values of GP and corresponding actions at Experience: {expr}")

    gp_fig.tight_layout()

    gp_plot_dir = os.path.join(plot_dir, 'GP/')
    if not os.path.isdir(gp_plot_dir):
        os.makedirs(gp_plot_dir)
    gp_fig.savefig(
        gp_plot_dir + f'GP_trajectories_{expr}.png', format='png')


def get_samples_and_log_prob(setup, policy, init_state_distn, fm, N_x0=10):
    x0s = init_state_distn.sample((N_x0,))
    fm_samples = x0s.new_empty(
        N_x0 * setup.N, setup.T, setup.state_dim)
    actions = x0s.new_empty(
        N_x0 * setup.N, setup.T, setup.action_dim)
    log_prob = fm_samples.new_zeros(N_x0 * setup.N)
    for j, x0 in enumerate(x0s):
        for i in range(setup.N):
            x = x0
            for t in range(setup.T):
                aux_x = convert_to_aux_state(x, setup.state_dim)
                x_t_u_t = torch.cat(
                    (aux_x, policy(aux_x).view(-1, setup.action_dim)), dim=1)
                y, log_prob_t = fm.predict(x_t_u_t, with_rsample=False,return_log_prob=True)
                log_prob[i + setup.N*j] += log_prob_t

                fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
                x = x + y.view(setup.state_dim)

                fm_samples[i + setup.N*j, t] = x
                actions[i + setup.N*j, t] = x_t_u_t[:, -
                                                    setup.action_dim:].detach().squeeze()
            fm.clear_fantasy_data()

    return fm_samples, log_prob, actions, x0s


def plot_progress(setup, expr, agent, plot_dir, policy, init_state_distn, fm, expert_samples, N_x0):
    """
    TODO: AT THE MOMENT STILL ASSUMES USING SCORE FUNCTION
    """
    # Plotting expert trajectory
    fig, axs = plot_trajectories(
        expert_samples, T=setup.T, color='blue', with_x0=False)

    # Plotting result of rollout with current policy
    for _ in range(20):
        with torch.no_grad():
            s_a_pairs, traj = agent.act()
        # Plot states
        for i, ax in enumerate(np.ravel(axs[:setup.state_dim])):
            ax.plot(np.arange(0, setup.T+1), traj.cpu().numpy()
                    [:, i], color='red', alpha=0.2)
        # Plot action
        axs[-1].plot(np.arange(0, setup.T), s_a_pairs[:, -
                                                1].cpu().numpy(), color='red', alpha=0.2)
    fig.suptitle(
        f"State values of expert vs learner with {expr} number of experience")

    fig.tight_layout()

    fig.savefig(
        plot_dir + f'expert_vs_learner_{expr}.png', format='png')

    # Plotting prediction of GP under current policy
    with torch.no_grad():
        samples, _, actions, x0s = get_samples_and_log_prob(
            setup, policy, init_state_distn, fm, N_x0=N_x0)
    plot_gp_trajectories(torch.cat((x0s.repeat_interleave(10, dim=0).unsqueeze(
        1), samples), dim=1), actions, T=setup.T, plot_dir=plot_dir, expr=expr)
