import numpy as np
import torch
import torch.distributions as trd


class Agent:
    def __init__(self, policy, T, dyn_std, start=0, device=torch.device('cpu')):
        """
        T is the maximum time, i.e. a trajectory would be list of T+1 positions
        """
        self.policy = policy
        self.T = T
        self.std = dyn_std
        self.start = start
        self.state_dim = 1
        self.device = device

        self.go_to_beginning()

    def _step(self):
        action = self.policy.action(self.curr)

        state_action = torch.cat((self.curr, action)).view(1, -1)
        if self.state_action_pairs is not None:
            self.state_action_pairs = torch.cat(
                (self.state_action_pairs, state_action))
        else:
            self.state_action_pairs = state_action

        # Note it is important here not to use += so that a new
        # object is created for self.curr each time
        self.curr = self.curr + action + self.std * \
            torch.randn(self.state_dim, device=self.device)

    def act(self):
        for _ in range(self.T):
            self._step()

    def get_state_action_pairs(self):
        """
        output: T x D
        """
        assert self.state_action_pairs is not None

        return self.state_action_pairs

    def get_curr_trajectory(self):
        """
        output: T+1
        """
        assert self.state_action_pairs is not None

        return torch.cat((self.state_action_pairs[:, 0], self.curr))

    def go_to_beginning(self):
        self.curr = self.start + self.std * \
            torch.randn(self.state_dim, device=self.device)
        # Would be a T+1 X D tensor
        self.state_action_pairs = None


class SimplePolicy:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.theta = torch.randn(1, requires_grad=True, device=device)

    def action(self, x):
        return self.theta

    def parameters(self):
        return [self.theta]

    def reset(self):
        self.theta = torch.randn(1, requires_grad=True, device=self.device)
