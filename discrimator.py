import torch
import torch.nn as nn
import torch.nn.functional as F


class Discrimator(nn.Module):
    def __init__(self, T, D=1):
        super().__init__()

        self.T = T
        self.D = D

        self.linear1 = nn.Linear(in_features=self.T*self.D,
                                 out_features=64, bias=True)

        self.linear2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, t):
        """
        t: batch_size x T x D
        """
        h = self.linear1(t.view(-1, self.T*self.D))
        return self.linear2(F.relu(h))
