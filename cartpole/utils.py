import torch
import pandas as pd
import os
import numpy as np
import matplotlib
font = {'size': 14}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt


def get_expert_data(N=64, clockwise=True):
    """
    output: N X T+1 x 4
    """
    result = torch.empty(N, 41, 4)

    orientation_folder = 'clockwise'
    if not clockwise:
        orientation_folder = 'anti-clockwise'

    for i in range(N):
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/expert_data/{orientation_folder}/data{i+1}.tsv', delimiter='\t')

        result[i, :, :] = torch.from_numpy(
            df.loc[:, ['x', 'v', 'dtheta', 'theta']].to_numpy())

    return result


def get_optimal_gp_inputs(with_theta=False):
    """
    output: 600 x (with_theta: 7, otherwise: 6)
    """
    df = pd.read_csv(
        f'{os.path.dirname(__file__)}/expert_data/optimal_gp_inputs.tsv', delimiter='\t')

    header = ['x', 'v', 'dtheta', 'theta', 'sin(theta)', 'cos(theta)', 'u']
    if not with_theta:
        header.remove('theta')
    return torch.from_numpy(df.loc[:, header].to_numpy())


def get_optimal_gp_targets():
    """
    output: 600 x 4
    """
    df = pd.read_csv(
        f'{os.path.dirname(__file__)}/expert_data/optimal_gp_targets.tsv', delimiter='\t')
    return torch.from_numpy(df.loc[:, ['x', 'v', 'dtheta', 'theta']].to_numpy())


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
    assert state.shape[-1] == D
    state = state.view(-1, D)

    return torch.cat((state[:, :D-1], torch.sin(state[:, D-1:D]), torch.cos(state[:, D-1:D])), dim=1)


def plot_trajectories(samples, T, actions=None, color=None, alpha=0.15, with_x0=True):
    """
    Creates a figure and axes for the states and actions.
    samples: N x T(+1) x D=4
    actions: N x T x 1, if None then empty plot added
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


def plot_gp_trajectories(fm_samples, actions, T, plot_dir, title, file_name, with_x0=True):
    gp_fig, _ = plot_trajectories(
        fm_samples, actions=actions, T=T, color='green', with_x0=with_x0)
    # gp_fig.suptitle(title)

    gp_fig.tight_layout()

    gp_plot_dir = os.path.join(plot_dir, 'GP/')
    if not os.path.isdir(gp_plot_dir):
        os.makedirs(gp_plot_dir)
    gp_fig.savefig(
        gp_plot_dir + f'{file_name}.pdf', format='pdf')
    plt.close(gp_fig)


def plot_progress(setup, expr, agent, plot_dir, policy, init_state_distn, fm, expert_samples, N_x0):
    """
    TODO: AT THE MOMENT STILL ASSUMES USING SCORE FUNCTION
    """
    # Plotting expert trajectory
    fig, axs = plot_trajectories(
        expert_samples, T=setup.T, color='blue', with_x0=expert_samples.shape[-2] == setup.T+1)

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
    # fig.suptitle(
        # f"State values of expert vs learner with {expr} number of experience")

    fig.tight_layout()

    fig.savefig(
        plot_dir + f'expert_vs_learner_{expr}.pdf', format='pdf')
    plt.close(fig)

    # Plotting prediction of GP under current policy
    with torch.no_grad():
        samples, actions = sample_trajectories(
            setup, fm, init_state_distn, policy, sample_N=20, sample_T=setup.T, with_rsample=False)
    plot_gp_trajectories(samples, actions, T=setup.T, plot_dir=plot_dir, title=f"Predicted state values of GP and corresponding actions after optimisation and Experience: {expr}", file_name=f'GP_trajectories_{expr}')


def sample_trajectories(setup, fm, init_state_distn, policy, sample_N, sample_T, with_rsample):
    """
    output:
        samples (x_1 to x_sample_T): sample_N, sample_T, setup.state_dim, 
        actions (u_0 to u_sample_T-1): sample_N, sample_T, setup.state_dim,
        x0s: sample_N, setup.state_dim
    """
    x0s = init_state_distn.sample((sample_N,))
    fm_samples = x0s.new_empty(sample_N, sample_T, setup.state_dim)
    actions = x0s.new_empty(sample_N, sample_T, setup.action_dim)

    for n, x in enumerate(x0s):
        for t in range(sample_T):
            aux_x = convert_to_aux_state(x, setup.state_dim)
            x_t_u_t = torch.cat(
                (aux_x, policy(aux_x).view(-1, setup.action_dim)), dim=1)
            y, _ = fm.predict(x_t_u_t, with_rsample=True)

            fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
            x = x + y.view(setup.state_dim)

            fm_samples[n, t] = x
            actions[n, t] = x_t_u_t[:, -setup.action_dim:].detach().squeeze()

            if t > 0 and t % 10 == 0:
                x = x.detach()

        fm.clear_fantasy_data()

    return torch.cat((x0s.unsqueeze(1), fm_samples), dim=1), actions


def save_current_state(expr, fm, disc, disc_optimizer, disc_dir, policy, policy_optimizer, policy_lr_sch, policy_dir):
    # Forward Model
    fm.save_training_data()
    fm.save_model_state()

    # Discriminator
    torch.save(disc.state_dict(), disc_dir+f'disc_after-expr-{expr}.pt')
    torch.save(disc_optimizer.state_dict(), disc_dir +
               f'optimizer_after_expr-{expr}.pt')

    # Policy
    torch.save(policy.state_dict(), policy_dir +
               f'policy_after_expr-{expr}.pt')
    torch.save(policy_optimizer.state_dict(),
               policy_dir+f'optimizer_after_expr-{expr}.pt')
    torch.save(policy_lr_sch.state_dict(), policy_dir +
               f'lr_scheduler_after_expr-{expr}.pt')


if __name__ == "__main__":
    # # Plot the optimal GP's training data
    # inputs = get_optimal_gp_inputs(with_theta=True).view(15, 40, 7)
    # samples = inputs[:, :, :4]
    # actions = inputs[:, :, -1].unsqueeze(-1)

    # plot_gp_trajectories(samples, actions[:, :39, :], T=39, plot_dir=os.path.dirname(
    #     __file__), title="Optimal GP training data", file_name='optimal_gp_data')


    # Plot clockwise expert trajectories
    samples = get_expert_data(clockwise=False)
    T = 40

    fig = plt.figure(figsize=(16, 10))
    axs = []
    state_action_names = [r'$x$', r'$v$',
                          r'$\dot{\theta}$', r'$\theta$']
    nrows = 2
    ncols = 2

    # Plot states
    for i in range(samples.shape[2]):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.plot(np.arange(0, T+1), torch.t(samples[:, :, i]).cpu(
        ).numpy(), color='blue', alpha=0.15)
        axs.append(ax)

    for i, ax in enumerate(axs):
        ax.set_xlabel("Time steps")
        ax.set_ylabel(state_action_names[i])

    fig.tight_layout()

    fig.savefig('cartpole/expert_data/anti-clockwise_expert_data.pdf', format='pdf')

    plt.close(fig)