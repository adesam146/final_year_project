import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDiscriminator(nn.Module):
    def __init__(self, D):
        super().__init__()

        self.D = D

        self.linear1 = nn.Linear(in_features=2*self.D,
                                 out_features=8, bias=True)

        self.linear2 = nn.Linear(in_features=8, out_features=1, bias=True)

    def forward(self, ss):
        """
        ss: batch_size x 2 x D
        First D columns corresponds to s_t and last to s_{t+1}
        """
        assert ss.shape[-1] == self.D
        assert ss.shape[-2] == 2

        h = self.linear1(ss.view(-1, 2 * self.D))
        return self.linear2(F.relu(h))

    def enable_parameters_grad(self, enable=True):
        for param in self.parameters():
            param.requires_grad = enable
