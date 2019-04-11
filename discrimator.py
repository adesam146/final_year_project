import torch
import torch.nn as nn
import torch.nn.functional as F


class Discrimator(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.T = T

        self.linear1 = nn.Linear(in_features=self.T+1,
                                 out_features=64, bias=True)

        self.linear2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, t):
        """
        t has shape (batch_size, T+1)
        """
        h = self.linear1(t)
        return self.linear2(F.relu(h))
