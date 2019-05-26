import torch
from cartpole.utils import convert_to_aux_state


def pathwise_grad(setup, expert_samples, policy, fm, disc,
                  disc_optimizer, policy_optimizer, init_state_distn):
    """
    output: discrimator loss, policy loss (both with grad detached)
    """

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    real_target = init_state_distn.mean.new_ones(setup.N, 1)
    fake_target = init_state_distn.mean.new_zeros(setup.N, 1)

    x0s = init_state_distn.sample((setup.N,))
    fm_samples = x0s.new_empty(setup.N, setup.T, setup.state_dim)
    actions = x0s.new_empty(setup.N, setup.T, setup.action_dim)

    for n, x in enumerate(x0s):
        for t in range(setup.T):
            aux_x = convert_to_aux_state(x, setup.state_dim)
            x_t_u_t = torch.cat(
                (aux_x, policy(aux_x).view(-1, setup.action_dim)), dim=1)
            y, _ = fm.predict(x_t_u_t, with_rsample=True)

            fm.add_fantasy_data(x_t_u_t.detach(), y.detach())
            x = x + y.view(setup.state_dim)

            fm_samples[n, t] = x
            actions[n, t] = x_t_u_t[:, -setup.action_dim:].detach().squeeze()
        fm.clear_fantasy_data()

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

    return disc_loss.detach(), policy_loss.detach()
