import numpy as np
import torch
import torch.distributions as trd
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, policy, T, dyn_std, start=0):
        """
        T is the maximum time, i.e. a trajectory would be list of T+1 positions
        """
        self.policy = policy
        self.T = T
        self.std = dyn_std
        self.trajectory = start + self.std * torch.randn(1)

    def _step(self):
        curr = self.trajectory[-1]
        action = self.policy.action(curr)
        # Note it is important here not to use += so that a new
        # object is created for self.curr each time
        curr = curr + action + self.std * torch.randn(1)
        self.trajectory = torch.cat((self.trajectory, curr))

    def act(self):
        for _ in range(self.T):
            self._step()

    def get_trajectory(self):
        """
        output: N x 1 tensor, where in this case 1 is the dim of the 
        states "x_t"
        """
        return self.trajectory.view(-1, 1)


class SimplePolicy:
    def __init__(self):
        self.theta = torch.randn(1, requires_grad=True)

    def action(self, x):
        return self.theta

    def parameters(self):
        return [self.theta]


policy = SimplePolicy()
agent = Agent(policy, 10, 0.01)

agent.act()

print(agent.get_trajectory())

agent.get_trajectory()[0].backward()
