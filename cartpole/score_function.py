import torch
import torch.nn.functional as F
from cartpole.utils import convert_to_aux_state, get_samples_and_log_prob


def score_function_training(setup, N_x0, expert_samples, policy, fm, disc, disc_optimizer, policy_optimizer, device, init_state_distn, policy_iter=50):
    """
    Optimize policy for given forward model using a GAN objective and score function gradients
    output: discrimator loss, policy loss (both with grad detached)
    """

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    real_target = torch.ones(setup.N, 1, device=device)
    fake_target = torch.zeros(setup.N*N_x0, 1, device=device)
    real_target_for_policy = torch.ones(setup.N*N_x0, 1, device=device)

    fm_samples, log_prob, _, _ = get_samples_and_log_prob(
        setup, policy, init_state_distn, fm, N_x0)

    # Train Discrimator
    disc.enable_parameters_grad()
    disc.train()
    disc_optimizer.zero_grad()

    # We detach forward model samples so that we don't calculate gradients w.r.t the policy parameter here
    disc_loss = bce_logit_loss(disc(fm_samples.detach(
    )), fake_target) + bce_logit_loss(disc(expert_samples), real_target)

    disc_loss.backward()
    disc_optimizer.step()

    # To avoid having to calculate gradients of discrimator
    disc.enable_parameters_grad(enable=False)
    disc.eval()

    # Optimise policy
    policy_optimizer.zero_grad()
    policy_loss = torch.mean(log_prob.view(-1, 1) * F.binary_cross_entropy_with_logits(
        disc(fm_samples.detach()), real_target_for_policy, reduction='none') - log_prob.view(-1, 1))
    policy_loss.backward()
    policy_optimizer.step()

    return disc_loss.detach(), policy_loss.detach()
