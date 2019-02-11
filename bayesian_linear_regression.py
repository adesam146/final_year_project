import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as trd
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions

# tf.enable_eager_execution()

# # scale is the std div
# beta = tfd.Normal(loc=0, scale=20)


# def trainable_normal(name=None):
#     with tf.variable_scope(None, default_name="trainable_normal"):
#         return tfd.Normal(loc=tf.get_variable("loc"), scale=tf.get_variable("scale"), name=name)


def linear_dataset(beta_true, N, noise_std=0.1):
    """
    For now just assuming that beta_true is a scalar
    """
    X = np.random.normal(loc=0, scale=1, size=(N, 1))
    noise = np.random.normal(loc=0, scale=noise_std, size=(N, 1))

    Y = beta_true * X + noise

    return X, Y


class TrainableNormal:
    def __init__(self, init_mean=0, init_std_dev=1):
        self.mean = torch.tensor(
            init_mean, requires_grad=True, dtype=torch.float)
        self.ln_sigma = torch.tensor(
            math.log(init_std_dev), requires_grad=True, dtype=torch.float)

    def log_prob(self, x):
        sigma2 = math.exp(2 * self.ln_sigma)

        return -0.5 * math.log(2*math.pi) - 0.5 * math.log(sigma2) - 0.5 * (1/sigma2) * (x - self.mean)**2

    def sample(self, batch_size=1):
        with torch.no_grad():
            output = self.mean + \
                math.exp(self.ln_sigma) * torch.randn(batch_size)

            return output

    def parameters(self):
        return [self.mean, self.ln_sigma]


class NormalLikelihoodSimulator():
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def simulate(self, beta, X):
        """
        X should be of the form (N, p)
        beta is of the form (p, 1)
        """
        mean = torch.matmul(X, beta)

        # output has shape (batch_size, N, 1)
        return mean + self.noise_std * torch.randn(X.shape[0])


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


def train_approx_posterior(prior, approx_posterior, ratio_estimator, data, posterior_optimizer):
    """
    Returns the posterior loss
    """
    beta_batch_size = 1
    posterior_optimizer.zero_grad()

    # Putting ration estimator in evaluation mode
    ratio_estimator.eval()
    expec_est_1 = 0
    for beta_sample in approx_posterior.sample(beta_batch_size):
        expec_est_1 += approx_posterior.log_prob(beta_sample) - \
            prior.log_prob(sample)

    sum_of_expec_est_2 = 0
    for X, Y in data:
        for beta_sample in approx_posterior.sample(beta_batch_size):
            sum_of_expec_est_2 += ratio_estimator(
                torch.cat(
                    beta_sample * torch.ones(X.shape[0]),
                    Y,
                    X)
                )

    loss = expec_est_1 + sum_of_expec_est_2

    loss.backward()

    posterior_optimizer.step()

    return loss


def train_ratio_estimator(beta, ratio_estimator, model_simulator, approx_simulator, data, ratio_optimizer):
    """
    Returns the ratio loss
    """
    X, Y = data

    # model_samples have shape (N, 1)
    model_samples = model_simulator.simulate(beta, X)

    if not approx_simulator:
        # This is for the trival case in which our approximate likelihood
        # is given by the emprical distribution

        approx_samples = Y

    ratio_estimator.train()

    # Prevent accumulation of grad for variables
    ratio_optimizer.zero_grad()

    # TODO: Calculate ratio loss

    model_estimates = ratio_estimator(
        torch.cat(
            (beta * torch.ones(X.shape[0]), model_samples, X),
            dim=1
        )
    )

    approx_estimates = ratio_estimator(
        torch.cat(
            (beta * torch.ones(X.shape[0]), approx_samples, X),
            dim=1
        )
    )

    ratio_loss = torch.mean(-F.logsigmoid(model_estimates)) + \
        torch.mean(-torch.log(1-F.sigmoid(approx_estimates)))

    ratio_loss.backward()

    ratio_optimizer.step()

    return ratio_loss


def inference(prior, approx_posterior, data_loader, model_simulator, approx_simulator, epochs=10):
    # The input features are beta, y, x
    ratio_estimator = RatioEstimator(in_features=3)
    ratio_optimizer = optim.SGD(ratio_estimator.parameters(), lr=0.01)

    posterior_optimizer = optim.SGD(approx_posterior.parameters(), lr=0.01)
    for epoch in range(epochs):
        for batch in data_loader:

            max_ratio_loss = -1
            for beta in approx_posterior.sample(batch_size=2):
                ratio_loss = train_ratio_estimator(
                    beta, ratio_estimator, model_simulator, approx_simulator, batch, ratio_optimizer)

                max_ratio_loss = max(max_ratio_loss, ratio_loss)

            posterior_loss = train_approx_posterior(
                prior, approx_posterior, ratio_estimator, batch, posterior_optimizer)

        print("Epoch: ", epoch, "max_ratio_loss:", max_ratio_loss,
              "posterior_loss:", posterior_loss)


# DATA
beta_true = np.array([10])
N_train = 1000
X_train, Y_train = linear_dataset(beta_true, N_train)

data_loader_train = DataLoader(TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=1)

# Model
prior = trd.Normal(0, 10)
model_simulator = NormalLikelihoodSimulator(noise_std=0.01)

# Approximation
approx_posterior = TrainableNormal()
approx_simulator = None

inference(prior, approx_posterior, data_loader_train,
          model_simulator, approx_simulator)
