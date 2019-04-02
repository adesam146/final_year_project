import numpy as np
import torch
import torch.distributions as trd
from matplotlib import pyplot as plt

class Agent:
  def __init__(self, T, dyn_std, start=0):
    """
    T is the maximum time, i.e. a trajectory would be list of T+1 positions
    """
    self.T = T
    self.std = dyn_std
    self.curr = start + self.std * np.random.randn()
    self.trajectory = [self.curr]

  def _step(self):
    #TODO: Mock
    action = 1
    self.curr += action + self.std * np.random.randn()
    self.trajectory.append(self.curr)

  def act(self):
    for _ in range(self.T):
      self._step()

  def get_trajectory(self):
    return torch.tensor(self.trajectory)


agent = Agent(10, 0.01)

agent.act()

print(agent.get_trajectory())
