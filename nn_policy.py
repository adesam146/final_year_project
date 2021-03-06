import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x):
    """
    Squashing the values in x to be between -1 and 1
    """
    return (9*torch.sin(x) + torch.sin(3*x))/8

class NNPolicy(nn.Module):
    def __init__(self, u_max, input_dim):
        super().__init__()

        self.u_max = u_max
        self.input_dim = input_dim
        self.linear1 = nn.Linear(in_features=self.input_dim,
                                 out_features=16, bias=True)
        self.linear2 = nn.Linear(in_features=16, out_features=1, bias=True)

    def __call__(self, x):
        """
        x: (1) x S
        output: 1
        """
        assert x.shape[-1] == self.input_dim

        h = self.linear1(x.view(-1, self.input_dim))
        return self.u_max * squash(self.linear2(F.relu(h))).view(1)

class DeepNNPolicy(nn.Module):
    def __init__(self, u_max, input_dim):
        super().__init__()

        self.u_max = u_max
        self.input_dim = input_dim
        self.linear1 = nn.Linear(in_features=self.input_dim,
                                 out_features=32, bias=True)
        self.linear2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.linear3 = nn.Linear(in_features=16, out_features=1, bias=True)

    def __call__(self, x):
        """
        x: (1) x S
        output: 1
        """
        assert x.shape[-1] == self.input_dim

        h = self.linear1(x.view(-1, self.input_dim))
        return self.u_max * squash(self.linear3(F.relu(self.linear2(F.relu(h))))).view(1)

