from setup import Setup

class CartPoleSetup():
  def __init__(self, N, T):
    # UNCHANGING DEFAULTS
    self.state_dim = 4
    self.aux_state_dim = 5
    self.action_dim = 1
    self.dt = 0.1

    self.N = N
    self.T = T