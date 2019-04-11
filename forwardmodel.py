import numpy as np
import torch
import gpytorch
import torch.distributions as trd
from matplotlib import pyplot as plt
import matplotlib.colorbar as cbar

# Planning to use gpytorch since it integrates well
# with pytorch, is modular (so should be quite flexible) and
# as the advantage of making use of GPUs if needed.


class ForwardModel:
    def __init__(self, init_x, init_y):
        """
        This model assumes the states and actions have already been paired
        to form the input into the GP
        init_x: N x D
        init_y: N x S
        """
        S = init_y.shape[1]
        # train_x is now S x N x D which is what the GP expects
        self.train_x = init_x.unsqueeze(0).repeat(S, 1, 1)
        # train_y is now S x N
        self.train_y = torch.t(init_y)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=S)
        self.model = GPModel(self.train_x, self.train_y, self.likelihood)

    def predict(self, x):
        """
        x: (Tst) x D
        output: S
        """
        self.model.eval()
        self.likelihood.eval()
        self.model.freeze_parameters()

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # We squeeze since output without would be: n x S x Tst = 1 x 1 x 1 in this case. Where n is number of sample and Tst is number of test points
        output = self.likelihood(self.model(x.view(-1, x.shape[-1]))).rsample().view(1)

        self.model.unfreeze_parameters()

        return output

    def __mean(self, x):
        """
        x: (Tst) x D
        output: S
        """
        self.model.eval()
        self.likelihood.eval()

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # We squeeze since output without would be: n x S x Tst = 1 x 1 x 1 in this case. Where n is number of sample and Tst is number of test points
        return self.likelihood(self.model(x.view(-1, x.shape[-1]))).mean.view(1)


    def learn(self):
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

    def plot_fm_mean(self, T=10):
        nx = 10
        X = np.linspace(-(T+1), T+1, nx)
        U = np.linspace(-2, 2, nx)

        # Note number of test point is nx*nx Tst

        Y = np.zeros((nx, nx))
        for i, x in enumerate(X):
            for j, u in enumerate(U):
                Y[i,j] = self.__mean(torch.tensor([x, u])).item()

        # Converting to mesh form
        X, U = np.meshgrid(X, U)

        # PLOTTING 3D CURVE
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        from matplotlib import cm
        # # Plot the surface.
        ax.scatter(self.train_x[0,:,0].numpy(), self.train_x[0,:,1].numpy(), self.train_y[0].numpy(), label="Data")
        surf = ax.plot_surface(X, U, Y, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.set_xlabel("x_t")
        ax.set_ylabel("u")
        ax.set_zlabel("x_t+1")
        ax.legend()

        # fig, ax = plt.subplots()
        # CS = ax.contour(X, U, Y)
        # fig.colorbar(CS, ax=ax)

        plt.show()


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """
        Let S be the dimension of the functions output,
        D the dimension of the input and N the number of data.
        We would model the actual function with S separate/independant GPs 
        train_y: S x N
        train_x: S x N x D
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        num_output = train_y.shape[0]
        self.mean_module = gpytorch.means.ConstantMean(batch_size=num_output)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=num_output), batch_size=num_output)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def freeze_parameters(self):
        """
        This would also prevent grads from the likelihood parameters
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """
        This would also allow grads from the likelihood parameters
        """
        for param in self.parameters():
            param.requires_grad = True