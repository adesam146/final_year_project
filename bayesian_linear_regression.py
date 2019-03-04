import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as trd
from trainable import TrainableNormal


def linear_dataset(beta_true, N, noise_std=0.1):
    """
    For now just assuming that beta_true is a scalar
    """
    X = np.random.normal(loc=0, scale=1, size=(N, 1))
    noise = np.random.normal(loc=0, scale=noise_std, size=(N, 1))

    Y = beta_true * X + noise

    return X, Y


class NormalLikelihoodSimulator():
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def simulate(self, beta, X):
        """
        TODO: using batching
        X should be of the form (N, 1)
        beta is of the form (1, 1)
        output has shape (N, 1)
        """
        mean = X * beta

        return mean + self.noise_std * torch.randn(X.shape[0], 1)

    def log_prob(self, y, beta, x):
        return -0.5 * np.log(2*math.pi) - 0.5 * np.log(self.noise_std ** 2) - 0.5 * (1/(self.noise_std ** 2)) * ((y - x * beta)**2)


class RatioEstimator(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.in_features = in_features

        self.linear1 = nn.Linear(
            in_features=in_features, out_features=64, bias=True)

        self.linear2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, inputs):
        h = self.linear1(inputs.view(-1, self.in_features))
        return self.linear2(F.relu(h))


def train_ratio_estimator(beta, ratio_estimator, model_simulator, approx_simulator, data, ratio_optimizer):
    """
    Returns the ratio loss
    """
    ratio_estimator.train()
    # Prevent accumulation of grad for variables
    ratio_optimizer.zero_grad()

    X, Y = data

    # model_samples have shape (N, 1)
    model_samples = model_simulator.simulate(beta, X)

    if not approx_simulator:
        # This is for the trival case in which our approximate likelihood
        # is given by the emprical distribution
        approx_samples = Y

    # Calculate ratio loss
    model_estimates = ratio_estimator(
        torch.cat(
            (model_samples, beta * torch.ones((Y.shape[0], 1))),
            dim=1)
    )

    approx_estimates = ratio_estimator(
        torch.cat(
            (approx_samples, beta * torch.ones((Y.shape[0], 1))),
            dim=1)

    )

    ratio_loss = F.binary_cross_entropy_with_logits(model_estimates, torch.ones_like(
        model_estimates)) + F.binary_cross_entropy_with_logits(approx_estimates, torch.zeros_like(approx_estimates))

    ratio_loss.backward()
    ratio_optimizer.step()

    return ratio_loss.detach().item()


def train_approx_posterior(prior, approx_posterior, ratio_estimator, data, posterior_optimizer):
    """
    Returns the posterior loss
    """
    approx_posterior.train()
    posterior_optimizer.zero_grad()

    # Putting ration estimator in evaluation mode
    ratio_estimator.eval()

    beta_sample = approx_posterior.rsample()

    # TODO: Make better use of batching
    expec_est_1 = approx_posterior.log_prob(beta_sample) - \
        prior.log_prob(beta_sample)

    _, Y = data
    sum_of_expec_est_2 = torch.sum(
        ratio_estimator(
            torch.cat((Y, beta_sample * torch.ones((Y.shape[0], 1))),
                      dim=1)
        )
    )

    loss = expec_est_1 - sum_of_expec_est_2

    loss.backward()
    posterior_optimizer.step()

    return loss.detach().item()


def inference(prior, approx_posterior, data_loader, model_simulator, approx_simulator, epochs=10):
    # The input features are y, beta
    ratio_estimator = RatioEstimator(in_features=2)
    ratio_optimizer = optim.Adam(
        ratio_estimator.parameters(), lr=0.1)

    posterior_optimizer = optim.Adam(
        approx_posterior.parameters(), lr=0.1)

    ratio_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
        ratio_optimizer, (0.9)**(1/100))
    posterior_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
        posterior_optimizer, (0.9)**(1/100))

    for epoch in range(epochs):
        ratio_lr_sch.step()
        posterior_lr_sch.step()
        for batch in data_loader:
            for _ in range(5):
                beta_sample = approx_posterior.sample()
                ratio_loss = train_ratio_estimator(
                    beta_sample, ratio_estimator, model_simulator, approx_simulator, batch, ratio_optimizer)

            posterior_loss = train_approx_posterior(
                prior, approx_posterior, ratio_estimator, batch, posterior_optimizer)

        print("Epoch: ", epoch, "ratio_loss:", ratio_loss,
              "post_loss:", posterior_loss)
        print("post mean", approx_posterior.mean, "post std_div",
              torch.exp(approx_posterior.ln_sigma))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # DATA
    beta_true = np.array([5])
    N_train = 500
    X_train, Y_train = linear_dataset(beta_true, N_train)

    train_dataset = TensorDataset(torch.from_numpy(
        X_train).float(), torch.from_numpy(Y_train).float())
    data_loader_train = DataLoader(train_dataset, batch_size=N_train)

    # Model
    m0 = 0
    S0 = 10 ** 2
    prior = trd.Normal(m0, np.sqrt(S0))
    noise_std = 1
    model_simulator = NormalLikelihoodSimulator(noise_std)

    # Approximation
    approx_posterior = TrainableNormal(init_mean=1)
    approx_simulator = None

    inference(prior, approx_posterior, data_loader_train,
              model_simulator, approx_simulator, epochs=1000)

    approx_posterior.eval()
    print("Learnt mean", approx_posterior.mean)
    print("Learnt std_dev", torch.exp(approx_posterior.ln_sigma))

    SN = 1/(1/S0 + 1/(noise_std**2) * np.dot(X_train.T, X_train))
    mN = SN * (1/S0 * m0 + 1/(noise_std**2) * np.dot(X_train.T, Y_train))
    print("Expected mean", mN)
    print("Expected std_div", np.sqrt(SN))
