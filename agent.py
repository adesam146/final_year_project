import numpy as np
import torch
import torch.distributions as trd
from matplotlib import pyplot as plt
import matplotlib.colorbar as cbar


class Agent:
    def __init__(self, policy, T, dyn_std, start=0):
        """
        T is the maximum time, i.e. a trajectory would be list of T+1 positions
        """
        self.policy = policy
        self.T = T
        self.std = dyn_std
        self.state_dim = 1
        self.curr = start + self.std * torch.randn(self.state_dim)
        # Would be a T X D tensor
        self.state_action_pairs = None


    def _step(self):
        action = self.policy.action(self.curr)

        state_action = torch.cat((self.curr, action)).view(1, -1)
        if self.state_action_pairs is not None:
            self.state_action_pairs = torch.cat(
                (self.state_action_pairs, state_action))
        else:
            self.state_action_pairs = state_action

        # Note it is important here not to use += so that a new
        # object is created for self.curr each time
        self.curr = self.curr + action + self.std * torch.randn(1)

    def act(self):
        for _ in range(self.T):
            self._step()

    def get_state_action_pairs(self):
        """
        Returns a T x D tensor
        """
        return self.state_action_pairs


class SimplePolicy:
    def __init__(self):
        self.theta = torch.randn(1, requires_grad=True)

    def action(self, x):
        return self.theta

    def parameters(self):
        return [self.theta]

if __name__ == "__main__":
    policy = SimplePolicy()
    T=10
    agent = Agent(policy, T, dyn_std=0.01)

    with torch.no_grad():
        # Only when generating samples from GP posterior do we need the grad wrt policy parameter
        agent.act()

    raw_pairs = agent.get_state_action_pairs() 
    init_x = raw_pairs[:-1]
    init_y = raw_pairs[1:, 0].reshape(-1, 1)
    print(init_x)
    print(init_y)

    from forwardmodel import ForwardModel

    fm = ForwardModel(init_x, init_y)

    fm.train()

    nx = 10
    X = np.linspace(-(T+1), T+1, nx)
    U = np.linspace(-2, 2, nx)

    # Note number of test point is nx*nx Tst

    Y = np.zeros((nx, nx))
    for i, x in enumerate(X):
        for j, u in enumerate(U):
            Y[i,j] = fm.mean(torch.tensor([x, u])).item()

    # Converting to mesh form
    X, U = np.meshgrid(X, U)

    # PLOTTING 3D CURVE
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    from matplotlib import cm
    # # Plot the surface.
    ax.scatter(init_x[:,0].numpy(), init_x[:,1].numpy(), init_y.numpy(), label="Data")
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





