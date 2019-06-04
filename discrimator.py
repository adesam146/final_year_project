import torch
import torch.nn as nn
import torch.nn.functional as F


class Discrimator(nn.Module):
    def __init__(self, T, D=1):
        super().__init__()

        self.T = T
        self.D = D

        self.linear1 = nn.Linear(in_features=self.T*self.D,
                                 out_features=16, bias=True)

        self.linear2 = nn.Linear(in_features=16, out_features=1, bias=True)

    def forward(self, t):
        """
        t: batch_size x T x D
        """
        assert t.shape[-2] == self.T
        assert t.shape[-1] == self.D
        h = self.linear1(t.view(-1, self.T*self.D))
        return self.linear2(F.leaky_relu(h))

    def enable_parameters_grad(self, enable=True):
        for param in self.parameters():
            param.requires_grad = enable


class ConvDiscrimator(nn.Module):
    def __init__(self, T, D):
        super().__init__()

        self.T = T
        self.D = D

        self.f_map_dim = 4
        self.conv1 = nn.Conv2d(1, self.f_map_dim, kernel_size=(
            2, self.D), stride=1, padding=0)
        self.linear1 = nn.Linear(self.f_map_dim*(self.T-1), 8, bias=True)
        self.linear2 = nn.Linear(8, 1, bias=True)

    def forward(self, t):
        """
        t: batch_size x T x D
        """
        assert t.shape[-2] == self.T
        assert t.shape[-1] == self.D

        h = F.leaky_relu(self.conv1(t.unsqueeze(1))).view(-1, self.f_map_dim*(self.T-1))
        return self.linear2(F.leaky_relu(self.linear1(h)))

    def enable_parameters_grad(self, enable=True):
        for param in self.parameters():
            param.requires_grad = enable

