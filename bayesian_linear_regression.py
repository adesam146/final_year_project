import math
import numpy as np
import torch
import torch.distributions as trd
from torch.utils.data import TensorDataset, DataLoader
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
        self.mean = torch.tensor(init_mean, requires_grad=True)
        self.ln_sigma = torch.tensor(
            math.log(init_std_dev), requires_grad=True)

    def log_prob(self, x):
        sigma2 = math.exp(2 * self.ln_sigma)

        return -0.5 * math.log(2*math.pi) - 0.5 * math.log(sigma2) - 0.5 * (1/sigma2) * (x - self.mean)**2

    def sample(self, n=1):
        with torch.no_grad():
            return self.mean + math.exp(self.ln_sigma) * torch.randn(n)


prior = trd.Normal(0, 10)
posterior = TrainableNormal()
batch = 10


def train_approx_posterior(prior, approx_posterior, ratio_estimator, data):
    # Putting ration estimator in evaluation mode
    ratio_estimator.eval()
    expec_est_1 = 0
    for sample in approx_posterior.sample(batch):
        expec_est_1 += approx_posterior.log_prob(sample) - \
            prior.log_prob(sample)

    sum_of_expec_est_2 = 0
    for x in data:
        for sample in approx_posterior.sample(batch):
            sum_of_expec_est_2 += ratio_estimator(x, sample)

    loss = expec_est_1 + sum_of_expec_est_2

    return loss


def train_ratio_estimator(ratio_estimator, model_simulator, approx_simulator):
    return None


class RatioEstimator():
    pass


def inference(prior, approx_posterior, data_loader, model_simulator, approx_simulator, epochs=10):
    ratio_estimator = RatioEstimator()
    ratio_optimizer = None
    posterior_optimizer = None
    for epoch in range(epochs):
        for batch in data_loader:
            ratio_loss = train_ratio_estimator(
                ratio_estimator, model_simulator, approx_simulator)

            # Prevent accumulation of grad for variables
            ratio_optimizer.zero_grad()
            ratio_loss.backward()
            ratio_optimizer.step()

            posterior_loss = train_approx_posterior(
                prior, approx_posterior, ratio_estimator, batch)

            posterior_optimizer.zero_grad()
            posterior_loss.backward()
            posterior_optimizer.step()

        print("Epoch: ", epoch, "ratio_loss:", ratio_loss, "posterior_loss:", posterior_loss)

beta_true=10
N_train = 1000

X_train, Y_train = linear_dataset(beta_true, N_train)
data_loader_train = DataLoader(TensorDataset(X_train, Y_train), batch_size=1)

inference(prior, posterior, data_loader_train, None, None)
