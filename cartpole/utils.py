import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def get_expert_data(N=64):
    """
    output: N X T x 4
    """
    result = torch.empty(N, 40, 4)

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
