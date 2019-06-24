import torch
from cartpole.utils import sample_trajectories


def state_to_state(setup, expert_samples, policy, fm, disc, disc_optimizer, policy_optimizer, init_state_distn):
    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    real_target = init_state_distn.mean.new_ones(
        setup.N, 1)
    fake_target = init_state_distn.mean.new_zeros(
        setup.N, 1)

    samples, actions = sample_trajectories(
        setup, fm, init_state_distn, policy, sample_N=setup.N, sample_T=setup.T, with_rsample=True)

    # Train Discriminator
    disc.enable_parameters_grad()
    disc_optimizer.zero_grad()

    disc_loss = 0
    for t in range(setup.T):
        disc_loss += bce_logit_loss(disc(expert_samples[:, t:t+2]), real_target) + bce_logit_loss(
            disc(samples[:, t:t+2].detach()), fake_target)

    disc_loss *= 1.0/setup.T

    disc_loss.backward()
    disc_optimizer.step()

    # Optimise policy
    policy_optimizer.zero_grad()

    # To avoid having to calculate gradients of discriminator
    disc.enable_parameters_grad(enable=False)

    policy_loss = 0
    for t in range(setup.T):
        policy_loss += bce_logit_loss(
            disc(samples[:, t:t+2]), real_target)

    policy_loss.backward()
    policy_optimizer.step()

    return policy_loss.detach(), disc_loss.detach(), samples.detach(), actions.detach()
