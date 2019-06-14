import torch
from cartpole.utils import convert_to_aux_state, sample_trajectories


def pathwise_grad(setup, expert_samples, policy, fm, disc,
                  disc_optimizer, policy_optimizer, init_state_distn):
    """
    output:
        discrimator loss (grad detached),
        policy loss (grad detached),
        samples: setup.N, setup.T+1, setup.state_dim
        actions: setup.N, setup.T, setup.action_dim
    """

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    real_target = init_state_distn.mean.new_ones(setup.N, 1)
    fake_target = init_state_distn.mean.new_zeros(setup.N, 1)

    fm_samples, actions = sample_trajectories(
        setup, fm, init_state_distn, policy, sample_N=setup.N, sample_T=setup.T, with_rsample=True)

    # Train Discrimator
    disc.enable_parameters_grad()
    disc_optimizer.zero_grad()

    # We detach forward model samples so that we don't calculate gradients w.r.t the policy parameter here
    disc_loss = bce_logit_loss(disc(fm_samples.detach(
    )), fake_target) + bce_logit_loss(disc(expert_samples), real_target)

    disc_loss.backward()
    disc_optimizer.step()

    # Optimise policy
    policy_optimizer.zero_grad()

    # To avoid having to calculate gradients of discrimator
    disc.enable_parameters_grad(enable=False)

    policy_loss = bce_logit_loss(disc(fm_samples), real_target)
    policy_loss.backward()
    policy_optimizer.step()

    return disc_loss.detach(), policy_loss.detach(), fm_samples.detach(), actions.detach()
