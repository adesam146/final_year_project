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
    def __init__(self, init_x, init_y, D, S, F, device):
        """
        This model assumes the states and actions have already been paired
        to form the input into the GP
        init_x: N x S+F
        init_y: N x D
        """
        assert init_x.shape[-1] == S+F
        assert init_y.shape[-1] == D
        self.D = D
        self.S = S
        self.F = F
        self.device = device

        self.train_x, self.train_y = self.__process_inputs(init_x, init_y)

        # See if this can be uncommented, I ideally want the same likelhood module throughout as I feel it allows reuse info?? Thereby, making finding optimal hyper-param easier
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=D)
        self.likelihood = None
        self.model = None

    def predict(self, x):
        """
        x: 1 x S+F
        sample: 1 x D
        log_prob: 1
        """
        assert self.model is not None
        assert x.shape[-1] == self.S + self.F

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # Output from sample is (n x ) D x Tst = (1 x) 1 x 1. Where n is number of sample and Tst is number of test points
        dyn_model = self.likelihood(self.model(self.__adjust_shape(x)))

        sample = dyn_model.sample()
        log_probs = dyn_model.log_prob(sample)

        return torch.t(sample), log_probs.sum() 

    def __mean(self, x):
        """
        x: (Tst) x S+F
        output: D
        """
        self.model.eval()
        self.likelihood.eval()

        # A model evaluated at x just returns a pytorch multivariate gaussian and calling the likelihood just transforms the distribution apporiately, i.e. at the noise variance
        # We squeeze since output without would be: n x D x Tst = 1 x 1 x 1 in this case. Where n is number of sample and Tst is number of test points
        return self.likelihood(self.model(self.__adjust_shape(x))).mean.view(1)

    def learn(self):
        if self.model is not None:
            # Consider how model can be set in constructor and reused
            # i.e. the fantasy model thing
            self.model.cpu()
            self.likelihood.cpu()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_size=self.D).to(self.device)
        self.model = GPModel(self.train_x, self.train_y,
                             self.likelihood).to(self.device)
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            # Includes GaussianLikelihood parameters
            {'params': self.model.parameters()},
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)

        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' %
                  (i + 1, training_iter, loss.item()))
            optimizer.step()

        self.model.freeze_parameters()
        self.model.eval()
        self.likelihood.eval()

        # TO be used for addition of fantasy data
        self.dummy_x = self.train_x
        self.dummy_y = self.train_y

        # Random computation for force the kernel to be evaluated
        # The reason this is done is that you do not want the first time the kernel for the data is calculated to be in a loop and involve a tensor that requires_grad because onces backward is called, in a subsequence iteration due to gpytorch's lazy loading, autograd would try to access the TODO
        self.predict(torch.zeros(1, self.S+self.F, device=self.device))

    def update_data(self, new_x, new_y):
        """
        new_x: N x S+F
        new_y: N x D
        """
        new_x, new_y = self.__process_inputs(new_x, new_y)

        self.train_x = torch.cat((self.train_x, new_x), dim=1)
        self.train_y = torch.cat((self.train_y, new_y), dim=1)

    def add_fantasy_data(self, x, y):
        """
        x: (N x) S+F
        y: (N x) D
        """
        assert self.model is not None
        assert x.shape[-1] == self.S+self.F
        assert y.shape[-1] == self.D

        x, y = self.__process_inputs(x.view(-1, self.S+self.F), y.view(-1, self.D))

        self.dummy_x = torch.cat((self.dummy_x, x), dim=1)
        self.dummy_y = torch.cat((self.dummy_y, y), dim=1)

        self.model.set_train_data(self.dummy_x, self.dummy_y, strict=False)

    def clear_fantasy_data(self):
        self.dummy_x = self.train_x
        self.dummy_y = self.train_y

    def __process_inputs(self, x, y):
        """
        Puts inputs in batch form for the Batch GP
        x: N x S+F
        y: N x D
        output[0]: D x N x S+F
        output[1]: D x N
        """
        x = self.__adjust_shape(x)
        # x is now D x N x S+F which is what the GP expects
        y = torch.t(y)
        # y is now D x N

        return x, y

    def __adjust_shape(self, x):
        """
        x: Tst x S+F
        output: D x Tst x S+F
        Need to replicate x for each target dimension
        """
        return x.unsqueeze(0).repeat(self.D, 1, 1)

    def plot_fm_mean(self, T=10):
        nx = 10
        X = np.linspace(-(T+1), T+1, nx)
        U = np.linspace(-2, 2, nx)

        # Note number of test point is nx*nx Tst

        Y = np.zeros((nx, nx))
        for i, x in enumerate(X):
            for j, u in enumerate(U):
                Y[i, j] = self.__mean(torch.tensor([x, u])).item()

        # Converting to mesh form
        X, U = np.meshgrid(X, U)

        # PLOTTING 3D CURVE
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        from matplotlib import cm
        # # Plot the surface.
        ax.scatter(self.train_x[0, :, 0].numpy(), self.train_x[0, :, 1].numpy(
        ), self.train_y[0].numpy(), label="Data")
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
        Let D be the dimension of the functions output,
        S+F the dimension of the input and N the number of data.
        We would model the actual function with D separate/independant GPs 
        train_y: D x N
        train_x: D x N x S+F
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
