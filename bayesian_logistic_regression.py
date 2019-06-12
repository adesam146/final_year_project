import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as trd
from torch.distributions import transform_to
from bayesian_linear_regression import inference, RatioEstimator
from trainable import TrainableMultivariateNormal
from matplotlib import pyplot as plt


def load_data(N=50, jitter=0.7, offset=1.2):
    """
    output:
        x_train: N x 2
        y_train: N x 1
        x_test: N x 2
    """
    # Generate the data
    x = np.vstack([np.random.normal(0, jitter, (N // 2, 1)),
                   np.random.normal(offset, jitter, (N // 2, 1))])
    # x: N x 1
    y = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    # y: N x 1
    # x_test = np.linspace(-2, offset + 2, num=N).reshape(-1, 1)
    # x_test: N x 1

    # Make the augmented data matrix by adding a column of ones
    x_train = np.hstack([np.ones((N, 1)), x])
    # x_train: N x 2

    x_test = np.vstack([np.random.normal(0, jitter, (N // 2, 1)),
                   np.random.normal(offset, jitter, (N // 2, 1))])
    x_test = np.hstack([np.ones((N, 1)), x_test])
    y_test = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    # x_test: N x 2
    return x_train, y, x_test, y_test


class BernoulliLikelihoodSimulator:

    def simulate(self, beta, X):
        """
        beta should have shape (D, 1)
        X should have shape (N, D)
        """
        return trd.Bernoulli(torch.sigmoid(torch.matmul(X, beta))).sample()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # DATA
    N_train = 500
    X_train, Y_train, X_test, Y_test = load_data(N_train)
    train_dataset = TensorDataset(torch.from_numpy(
        X_train).float(), torch.from_numpy(Y_train).float())
    data_loader_train = DataLoader(train_dataset, batch_size=N_train)

    # Model
    D = X_train.shape[1]
    m0 = torch.zeros(D)
    S0 = 10 * torch.eye(D)
    prior = trd.MultivariateNormal(loc=m0, covariance_matrix=S0)
    model_simulator = BernoulliLikelihoodSimulator()

    # Approximation
    approx_posterior = TrainableMultivariateNormal(mean=torch.ones(D), cov=S0)
    approx_simulator = None

    ratio_estimator = RatioEstimator(in_features=1+D)

    inference(ratio_estimator, prior, approx_posterior, data_loader_train,
              model_simulator, approx_simulator, epochs=1000)

    print("Learnt mean", approx_posterior.mean)
    print("Leant cov", approx_posterior.cov())

    beta_mean = approx_posterior.mean
    print(beta_mean)

    from sklearn import metrics

    # From Coursework
    N_m = 1000
    y_hat = 0
    for _ in range(N_m):
        y_hat += model_simulator.simulate(approx_posterior.sample().view(D, 1), torch.from_numpy(
            X_test).float()).numpy()

    y_hat *= 1.0/N_m  
    y_hat = y_hat.squeeze() > 0.5
    print("Accuracy:", metrics.accuracy_score(Y_test, y_hat))
    print("Confusion matrix", metrics.confusion_matrix(Y_test, y_hat))

    plt.scatter(X_test[y_hat < 0.5, 1], Y_test[y_hat < 0.5])
    plt.scatter(X_test[y_hat > 0.5, 1], Y_test[y_hat > 0.5])
    X = np.linspace(np.min(X_test[:, 1]), np.max(X_test[:, 1]))
    plt.plot(X, 1/(1+np.exp(-X)), color='black')
    plt.legend(['Sigmoid function', 'Classified as 0', 'Classified as 1'])
    # plt.plot(X_test[:, 1], model_simulator.simulate(
    # beta_mean, torch.from_numpy(X_test).float()).numpy())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
