import numpy as np
from scipy.integrate import odeint
import torch
import torch.distributions as trd
from cartpole.utils import convert_to_aux_state


class CartPoleAgent():
    def __init__(self, time, dt, policy, measurement_var, init_state_distn=None, device=torch.device('cpu')):
        """
        The states are
        1  x          cart position
        2  v          cart velocity
        3  dtheta     angular velocity
        4  theta      angle of the pendulum
        """
        self.state_dim = 4
        self.dt = dt
        self.T = int(np.ceil(time/self.dt))
        self.policy = policy
        self.device = device

        self.init_state_distn = init_state_distn
        if init_state_distn is None:
            self.init_state_distn = trd.MultivariateNormal(torch.zeros(
                self.state_dim, device=self.device), covariance_matrix=torch.diag(0.1**2*torch.ones(self.state_dim, device=self.device)))

        # This is the measurement noise
        self.err_distn = trd.MultivariateNormal(
            torch.zeros(self.state_dim, device=self.device), covariance_matrix=measurement_var.to(self.device))

        self.go_to_beginning()

    def step(self):
        # Assuming the cart is stationary and the pole is vertical and also stationary
        action = self.policy(convert_to_aux_state(
            self.state, D=self.state_dim))

        # Updating state action pair and adding measurement noise to state
        state_action = torch.cat(
            (self.state + self.err_distn.sample(), action)).view(1, -1)
        if self.state_action_pairs is not None:
            self.state_action_pairs = torch.cat(
                (self.state_action_pairs, state_action))
        else:
            self.state_action_pairs = state_action

        # Solving ODE
        # Using a zero order hold with no delay for the controller
        sol = odeint(cartpole_dynamics, self.state.cpu().numpy(), t=np.array(
            [0, self.dt]), args=(lambda t: action.detach().cpu().numpy().item(),), tfirst=True)

        self.state = torch.from_numpy(sol[1, :]).type(
            action.dtype).to(self.device)

    def act(self):
        """
        Returns:
        state_action_pairs: T x D+F
        trajectory: (T+1) x D 
        and resets agent to beginning
        """
        for _ in range(self.T):
            self.step()

        output = self.state_action_pairs, torch.cat(
            (self.state_action_pairs[:, :self.state_dim], self.state.view(1, -1)))

        self.go_to_beginning()

        return output

    def go_to_beginning(self):
        self.state = self.init_state_distn.sample()

        # Going to be a T x D+F
        self.state_action_pairs = None


def cartpole_dynamics(t, z, f):
    m1 = 0.5  # Cart mass kg
    m2 = 0.5  # Pole mass kg
    l = 0.5  # Pole length m
    b = 0.1  # Friction between cart and ground N/m/s
    g = 9.82  # Gravity m/s^2

    dz1 = z[1]
    dz2 = (2*m2*l*z[2]**2*np.sin(z[3]) + 3*m2*g*np.sin(z[3])*np.cos(z[3]
                                                                    ) + 4*f(t) - 4*b*z[1])/(4*(m1 + m2) - 3*m2*np.cos(z[3])**2)
    dz3 = (-3*m2*l*z[2]**2*np.sin(z[3])*np.cos(z[3]) - 6*(m1+m2)*g*np.sin(z[3]) -
           6*(f(t) - b*z[1])*np.cos(z[3]))/(4*l*(m1 + m2) - 3*m2*l*np.cos(z[3])**2)
    dz4 = z[2]

    return np.array([dz1, dz2, dz3, dz4])
