import numpy as np
import torch
import torch.distributions as trd


class Agent:
    def __init__(self, policy, T, dyn_std, start=0):
        """
        T is the maximum time, i.e. a trajectory would be list of T+1 positions
        """
        self.policy = policy
        self.T = T
        self.std = dyn_std
        self.start = start
        self.state_dim = 1
        self.curr = start + self.std * torch.randn(self.state_dim)
        # Would be a T+1 X D tensor
        self.state_action_pairs = None


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
        self.curr = self.curr + action + self.std * torch.randn(self.state_dim)

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


class SimplePolicy:
    def __init__(self):
        self.theta = torch.randn(1, requires_grad=True)

    def action(self, x):
        return self.theta

    def parameters(self):
        return [self.theta]






