import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as trd
from trainable import TrainableNormal, TrainableMultivariateNormal
from mixture_likelihood_simulator import MixtureLikelihoodSimulator

# *** ARGUMENT SET UP ***
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int,
                    help="Number of iteration/epoch for the posterior optimisation (default=2000)")
args = parser.parse_args()


def linear_dataset(beta_true, N, noise_std=0.1):
    """
    beta_true: D
    output: 
        X: N x D
        Y: N x 1
    """
    X = torch.randn(N, beta_true.shape[0])
    noise = noise_std * torch.randn(N, 1)

    Y = torch.matmul(X, beta_true.view(-1, 1)) + noise

    return X, Y

def pure_cubic_dataset(beta_true, N, noise_std=1):
    """
    For now just assuming that beta_true is a scalar
    """
    # X = torch.linspace(-5, 5, steps=N).view(N, 1)
    # X = torch.randn(N, 1)
    X = 4 * torch.rand(N, 1) - 2
    noise = noise_std * torch.randn(N, 1)

    Y = beta_true * X ** 3 + noise

    return X, Y

def pure_cubic_dataset_with_mixture_likelihood(beta_true, N, noise_std=1):
    """
    For now just assuming that beta_true is a scalar
    output: N x 1
    """
    X = 2 * torch.rand(N, 1)

    simulator = MixtureLikelihoodSimulator(noise_std=noise_std, convert_to_design_matrix=lambda X: X**3)

    return X, simulator.simulate(beta=beta_true, X=X)

def polynomial_basis_funcs(x, order):
    """
    x is assummed to be a single point i.e. in R while
    order is the order of the polynomials
    Return value: a numpy array of shape (order+1,)
    """
    result = [1]
    for i in range(1, order+1):
        result.append(x ** i)

    return np.array(result)


def polynomial_design_matrix(X, order):
    """
    X: N (x 1)
    order: >=0
    output: a numpy array of shape N x order+1
    """
    result = []
    for x_arr in X.view(-1, 1):
        result.append(polynomial_basis_funcs(x_arr[0], order))

    return np.array(result)


class NormalLikelihoodSimulator():
    def __init__(self, noise_std, convert_to_design_matrix):
        self.noise_std = noise_std
        self.convert_to_design_matrix = convert_to_design_matrix

    def simulate(self, beta, X):
        """
        TODO: using batching
        X should be of the form (N, D)
        beta is of the form (D, 1)
        output has shape (N, 1)
        """
        mean = torch.matmul(self.convert_to_design_matrix(X), beta.view(-1,1))

        return mean + self.noise_std * torch.randn(X.shape[0], 1)

    def log_prob(self, y, beta, x):
        # TODO: update to work with multi-dimensional beta
        return -0.5 * np.log(2*math.pi) - 0.5 * np.log(self.noise_std ** 2) - 0.5 * (1/(self.noise_std ** 2)) * ((y - x * beta)**2)


class RatioEstimator(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.in_features = in_features

        self.linear1 = nn.Linear(
            in_features=in_features, out_features=64, bias=True)

        self.linear2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, y, beta):
        """
        y has shape (batch_size, 1)
        beta has shape (1, D)
        """
        B = y.shape[0]
        D = beta.shape[0]
        inputs = torch.cat(
            (y.view(-1, 1), beta.view(1, D).expand(B, D)),
            dim=1)
        h = self.linear1(inputs.view(-1, self.in_features))
        return self.linear2(F.leaky_relu(h))


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
    model_estimates = ratio_estimator(model_samples, beta)

    approx_estimates = ratio_estimator(approx_samples, beta)

    ratio_loss = F.binary_cross_entropy_with_logits(model_estimates, torch.ones_like(
        model_estimates)) + F.binary_cross_entropy_with_logits(approx_estimates, torch.zeros_like(approx_estimates))

    ratio_loss.backward()
    ratio_optimizer.step()

    return ratio_loss.detach().item()


def train_approx_posterior(prior, approx_posterior, ratio_estimator, data, posterior_optimizer):
    """
    Returns the posterior loss
    """
    posterior_optimizer.zero_grad()

    # Putting ration estimator in evaluation mode if needed
    ratio_estimator.eval()

    beta_sample = approx_posterior.rsample()

    expec_est_1 = trd.kl.kl_divergence(trd.MultivariateNormal(loc=approx_posterior.mean, covariance_matrix=approx_posterior.cov()), prior)

    _, Y = data
    sum_of_expec_est_2 = torch.sum(ratio_estimator(Y, beta_sample))

    loss = expec_est_1 - sum_of_expec_est_2

    loss.backward()
    posterior_optimizer.step()

    return loss.detach().item()


def inference(ratio_estimator, prior, approx_posterior, data_loader, model_simulator, approx_simulator, epochs=10):
    ratio_optimizer = optim.Adam(
        ratio_estimator.parameters(), lr=0.1)

    posterior_optimizer = optim.Adam(
        approx_posterior.parameters(), lr=0.1)

    ratio_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
        ratio_optimizer, (0.9)**(1/100))
    posterior_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
        posterior_optimizer, (0.9)**(1/100))

    posterior_losses = []

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

        posterior_losses.append(posterior_loss)
        print("Epoch: ", epoch, "ratio_loss:", ratio_loss,
              "post_loss:", posterior_loss)
        print("post mean", approx_posterior.mean, "post variance",
              approx_posterior.cov())

    return np.array(posterior_losses)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_default_dtype(torch.float64)

    # DATA
    # N_train = 50
    N_train = 1000

    # 2D linear regression (i.e. X is 2D)
    # D = 2
    # beta_true = 5.0 * torch.ones(D)
    # X_train, Y_train = linear_dataset(beta_true, N_train)
    # convert_to_design_matrix = lambda X: X

    # Polynomial of order 3 with no intersect or quadratic term
    # D = 1
    # beta_true = 1.0 * torch.ones(D)
    # X_train, Y_train = pure_cubic_dataset(beta_true, N_train)
    # convert_to_design_matrix = lambda X: torch.tensor(polynomial_design_matrix(X, order=2))[:, -1:]

    D = 1
    beta_true = 1.0 * torch.ones(D)
    X_train, Y_train = pure_cubic_dataset_with_mixture_likelihood(beta_true, N_train)
    convert_to_design_matrix = lambda X: torch.tensor(polynomial_design_matrix(X, order=3))[:, -1:]

    # Polynomial of order 3
    # D = 4
    # X_train = torch.randn(N_train, 1)
    # beta_true = torch.tensor([4, 3, 2 , 1])
    # Y_train = 0
    # for i, coeff in enumerate(beta_true):
    #     Y_train += coeff * X_train ** i
    # Y_train += torch.randn(N_train, 1)
    # convert_to_design_matrix = lambda X: torch.tensor(polynomial_design_matrix(X, order=3))

    # Polynomial of order 3
    # D = 3
    # X_train = torch.randn(N_train, 1)
    # beta_true = torch.tensor([3, 2 , 1])
    # Y_train = 0
    # for i, coeff in enumerate(beta_true):
    #     Y_train += coeff * X_train ** i
    # Y_train += torch.randn(N_train, 1)
    # convert_to_design_matrix = lambda X: torch.tensor(polynomial_design_matrix(X, order=2))

    train_dataset = TensorDataset(X_train, Y_train)
    data_loader_train = DataLoader(train_dataset, batch_size=N_train)

    # Model
    m0 = torch.zeros(D)
    S0 = 36 * torch.eye(D)
    prior = trd.MultivariateNormal(loc=m0, covariance_matrix=S0)
    noise_std = 1
    # model_simulator = NormalLikelihoodSimulator(noise_std, convert_to_design_matrix=convert_to_design_matrix)
    model_simulator= MixtureLikelihoodSimulator(noise_std, convert_to_design_matrix=convert_to_design_matrix)

    # Approximation
    approx_posterior = TrainableMultivariateNormal(mean=torch.zeros(D) + 2, cov=torch.eye(D))
    approx_simulator = None

    # The input features are y, beta
    ratio_estimator = RatioEstimator(in_features=2)

    posterior_losses = inference(ratio_estimator, prior, approx_posterior, data_loader_train,
              model_simulator, approx_simulator, epochs=args.iter or 2000)

    # The learnt std_dev tends to be larger than the expected and this
    # is also the case for the implementation in Edward (original)
    # Furthermore both in this and the original after more than 1000
    # iteration the learnt mean goes above the expected with more than
    # a decimal place difference
    print("Learnt mean", approx_posterior.mean)
    print("Learnt variance", approx_posterior.cov())

    Phi = convert_to_design_matrix(X_train)
    SN = torch.inverse(torch.inverse(S0) + 1/(noise_std**2) * torch.matmul(Phi.t(), Phi))
    mN = torch.matmul(SN, torch.matmul(torch.inverse(S0), m0) + 1/(noise_std**2) * torch.matmul(Phi.t(), Y_train.squeeze()))
    print("Expected mean", mN)
    print("Expected variance", SN)

    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)

    X = X_train.squeeze().numpy()
    order = np.argsort(X)
    X = X[order]
    Y = Y_train.squeeze().numpy()[order]

    mean_pred = torch.matmul(Phi, approx_posterior.mean.detach()).squeeze().numpy()[order]
    std_pred = torch.sqrt(torch.diag(torch.chain_matmul(Phi, approx_posterior.cov().detach(), Phi.t()))).numpy()[order]

    mean_pred_exact = torch.matmul(Phi, mN).squeeze().numpy()[order]
    std_pred_exact = torch.sqrt(torch.diag(torch.chain_matmul(Phi, SN, Phi.t()))).numpy()[order]


    ax.set_ylim(bottom=-10, top=10)
    ax.set_ylim(bottom=-10, top=10)

    ax.scatter(X, Y, label='Observed Y', color='black', s=1, alpha=0.5)
    # ax.fill_between(X, Y - (X ** 3) * np.sqrt(SN.item()), Y + (X ** 3) * np.sqrt(SN.item()), alpha = 0.15)

    ax.plot(X, mean_pred, label='Mean Prediction of LFVI')
    ax.fill_between(X, mean_pred - 2 * std_pred, mean_pred + 2 * std_pred, alpha=0.15)

    ax.plot(X, mean_pred_exact, label='Mean prediction of exact BLR')
    ax.fill_between(X, mean_pred_exact - 2 * std_pred_exact, mean_pred_exact + 2 * std_pred_exact, alpha=0.15)

    ax.plot(X, X**3, label=r'True function $y = x^3$')

    ax.legend(loc='upper left')

    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')

    plt.show()

    # plt.plot(posterior_losses)
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Posterior loss of LFVI")
    # plt.show()