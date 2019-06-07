import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed to ensure that results are reproducible.
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)


def load_data(N=100):
    """
    output:
      train_x: N x 1
      train_y: N x 1
    """
    X = 2 * torch.randn(N, 1)

    # x^2 + 2x + 1
    Y = (X + 1)**2 + 0.01 * torch.randn(N, 1)

    return X, Y


class Function(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features=2,
                                 out_features=4, bias=True)

        self.linear2 = nn.Linear(in_features=4, out_features=1, bias=True)

    def forward(self, f_x, ep):
        """
        f_x: (batch_size x) 1
        ep: (batch_size x) 1
        """
        assert f_x.shape[-1] == 1

        h = torch.cat((f_x.view(-1, 1), ep.view(-1, 1)), dim=1)
        return self.linear2(F.relu(self.linear1(h)))


class Discriminator(nn.Module):
    def __init__(self, x_dim=1, y_dim=1):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.linear1 = nn.Linear(in_features=self.x_dim+self.y_dim,
                                 out_features=4, bias=True)

        self.linear2 = nn.Linear(in_features=4, out_features=1, bias=True)

    def forward(self, x, y):
        """
        x: (batch_size x) x_dim
        y: (batch_size x) y_dim
        """
        assert x.shape[-1] == self.x_dim
        assert y.shape[-1] == self.y_dim

        h = torch.cat((x.view(-1, self.x_dim), y.view(-1, self.y_dim)), dim=1)
        return self.linear2(F.relu(self.linear1(h)))

if __name__ == "__main__":
    N = 1000
    train_X, train_Y = load_data(N=N)

    func = Function()
    func_optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)

    gen = Generator()
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-2)

    disc = Discriminator()
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-2)

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    real_target = torch.ones(N, 1)
    fake_target = torch.zeros(N, 1)

    epochs = 2000
    for epoch in range(epochs):

        eps = torch.randn_like(train_Y)
        fake_Y = gen(func(train_X), eps)

        # Optimise discrimator
        disc_optimizer.zero_grad()

        disc_loss = bce_logit_loss(disc(train_X, train_Y), real_target) + \
            bce_logit_loss(disc(train_X, fake_Y.detach()), fake_target)

        disc_loss.backward()
        disc_optimizer.step()

        # Optimise generator and function
        gen_optimizer.zero_grad()
        func_optimizer.zero_grad()

        model_loss = bce_logit_loss(disc(train_X, fake_Y), real_target)
        model_loss.backward()

        gen_optimizer.step()
        func_optimizer.step()

        print(
            f'Epoch {epoch}, disc loss: {disc_loss.detach()}, model loss: {model_loss.detach()}')

    print(f'a: {func.a.item()}, b: {func.b.item()}, c:{func.c.item()}')

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(train_X.squeeze().numpy(), train_Y.squeeze().numpy(), label='Observed Y')
    ax.scatter(train_X.squeeze().numpy(), func(train_X).detach().squeeze().numpy(), label=r"Learnt function: $f_{\theta_m^*}(x)$")
    ax.scatter(train_X.squeeze().numpy(), gen(func(train_X), torch.randn_like(train_Y)).detach().squeeze().numpy(), label=r"Generator applied to learnt function: $G_{\theta_g^*}(f_{\theta_m^*}(x), \epsilon)$")

    ax.legend()

    fig.tight_layout()
    fig.savefig("gan_regression_failure.png", format='png')


    plt.close(fig)