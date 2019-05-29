import torch
import torch.nn.functional as F
from cartpole.utils import convert_to_aux_state


def score_function_training(setup, N_x0, expert_samples, policy, fm, disc, disc_optimizer, policy_optimizer, init_state_distn, policy_iter=50):
    """
    Optimize policy for given forward model using a GAN objective and score function gradients
    output: 
        discrimator loss (grad detached),
        policy loss (grad detached),
        samples: N_x0 * N, setup.T+1, setup.state_dim
        actions: N_x0 * N, setup.T, setup.action_dim
    """

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    real_target = init_state_distn.mean.new_ones(setup.N, 1)
    fake_target = init_state_distn.mean.new_zeros(setup.N*N_x0, 1)
    real_target_for_policy = init_state_distn.mean.new_ones(setup.N*N_x0, 1)

    fm_samples, log_prob, actions, x0s = get_samples_and_log_prob(
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

    return disc_loss.detach(), policy_loss.detach(), torch.cat((x0s.unsqueeze(1), fm_samples.detach()), dim=1), actions

def get_samples_and_log_prob(setup, policy, init_state_distn, fm, N_x0=10):
    """
    output:
      samples: N_x0 * setup.N, setup.T, setup.state_dim
      log_prob: N_x0 * setup.N
      actions (Detached): N_x0 * setup.N, setup.T, setup.action_dim
      x0s: N_x0 * setup.N, setup.state_dim
    """
    x0s = init_state_distn.sample((N_x0,))
    fm_samples = x0s.new_empty(
        N_x0 * setup.N, setup.T, setup.state_dim)
    actions = x0s.new_empty(
        N_x0 * setup.N, setup.T, setup.action_dim)
    log_prob = fm_samples.new_zeros(N_x0 * setup.N)
    for j, x0 in enumerate(x0s):
        print(f'j: {j}')
        for i in range(setup.N):
            x = x0
            for t in range(setup.T):
                aux_x = convert_to_aux_state(x, setup.state_dim)
                x_t_u_t = torch.cat(
                    (aux_x, policy(aux_x).view(-1, setup.action_dim)), dim=1)
                y, log_prob_t = fm.predict(
                    x_t_u_t, with_rsample=False, return_log_prob=True)
                log_prob[i + setup.N*j] += log_prob_t

                fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
                x = x + y.view(setup.state_dim)

                fm_samples[i + setup.N*j, t] = x
                actions[i + setup.N*j, t] = x_t_u_t[:, -
                                                    setup.action_dim:].detach().squeeze()
            fm.clear_fantasy_data()

    return fm_samples, log_prob, actions.detach(), x0s.detach().repeat_interleave(setup.N, dim=0)
