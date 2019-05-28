import torch


class OptimalPolicy:
    # Consider making an nn module
    def __init__(self, u_max, device):
        self.u_max = u_max
        self.device = device

        self.X = torch.tensor([[-4.13779749423284,	0.388195736104642,	0.311289659549716,	-2.16182462811072,	0.987866588280126],
                               [3.78758225404730,	-1.50196979968876,
                                2.29029320172610,	-1.95086857483679,	0.117002387329019],
                               [0.652560883513185,	0.327118752333231,	0.155059137379251,
                                2.50204549380119,	8.37968894552629],
                               [1.72112825674993, -1.53247547699814,	2.35186941629716,	-
                                1.98119354870294, 0.108531762035523],
                               [-0.702266482536873,	0.952642018591305,	-
                                0.997256324616101,	-2.06323890705037,	0.691691983630037],
                               [7.43782550167155,	-0.196633634525162,	-2.29115509596089,
                                0.0158841957305422,	-0.479030747258707],
                               [-7.46563182842172,	-0.424185183866230,	-
                                1.64166730971484,	-0.419958386283865,	-0.455840147146565],
                               [-2.53741650954214,	1.33442324932381,	-
                                2.12971310652922,	-1.96894073012227,	0.452319836658442],
                               [-1.94021641585795,	2.08713664072762,	-5.10702508534269,	-
                                2.09700193997521,	-0.147539886461367],
                               [-2.29523332434410,	0.587459048200402,	-0.127238083265639,	-2.12728785878586,	0.887069801692657]], requires_grad=True, device=self.device)

        self.W = torch.diag(torch.exp( -2 * torch.tensor([2.34165811758010,
                                          0.652757197256950,
                                          1.35740406373520,
                                          0.520681839124039,
                                          -0.309966200635860], device=self.device)))
        self.W.requires_grad = True
        self.Y = torch.tensor([-14.7283720839688,
                               -19.6035046557288,
                               -2.35525991310959,
                               17.8296433086668,
                               -9.79267204942738,
                               16.6193178641639,
                               -11.4750931214345,
                               6.33739805806107,
                               -1.51330986923437,
                               -0.589476409536458], requires_grad=True, device=self.device)

        self.recompute_K()


    def recompute_K(self):
        self.K = torch.zeros(10, 10, device=self.device)
        for i in range(10):
            for j in range(10):
                self.K[i, j] = self.kernel(self.X[i], self.X[j])
                if i == j:
                    self.K[i, j] += 0.01**2


    def __call__(self, x):
        self.recompute_K()

        k = torch.zeros(10, device=self.device)

        for i in range(10):
            k[i] = self.kernel(x, self.X[i])

        l = torch.cholesky(self.K)

        v = torch.cholesky_solve(self.Y.view(-1, 1), l)
        
        return self.u_max * self.squash(torch.matmul(k.view(1, -1), v)).view(1)

    def kernel(self, x1, x2):
        """
        x1, x2: 5
        """
        return torch.exp(-0.5 * torch.chain_matmul((x1-x2).view(1, 5), self.W, (x1-x2).view(5, 1)))

    def squash(self, x):
        """
        Squashing the values in x to be between -1 and 1
        """
        return (9*torch.sin(x) + torch.sin(3*x))/8

    def parameters(self):
        # return [self.weights, self.centers, self.ln_vars]
        return [self.W, self.Y, self.X]

    def eval(self):
        pass

    def train(self):
        pass

if __name__ == "__main__":
    print("TESTING OPTIMAL POLICY")
    import numpy as np
    torch.set_default_dtype(torch.float64)

    # Testing Policy, TODO: Convert to test
    policy = OptimalPolicy(10, input_dim=1, nbasis=2, device=torch.device('cpu'))

    # Testing against output from optimal policy in matlab
    # with the optimal parameters hardcorded into policy.
    test1 = policy(torch.tensor([0, 0, 0, 0, 1.0]))
    np.testing.assert_almost_equal(test1.detach().numpy(), np.array([3.771637760361378]), decimal=4)

    test2 = policy(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(test2.detach().numpy(), np.array([-1.417889544030735]), decimal=4)

    test3 = policy(torch.tensor([0.5, 1, 1.5, 2, 2.5]))
    np.testing.assert_almost_equal(test3.detach().numpy(), np.array([-3.679293910231314]), decimal=4)

    print("OPTIMAL POLICY TEST SUCCESSFULL")

    # print("Weights", policy.weights)
    # unsquashed = policy.weights[0] * torch.exp(-0.5 * policy.centers[0, 0]**2 * 1/torch.exp(policy.ln_vars)) + policy.weights[1] * torch.exp(-0.5 * policy.centers[0, 1]**2 * 1/torch.exp(policy.ln_vars))
    # print("Unsquashed", unsquashed)
    # print("Squashed", policy.squash(unsquashed))
    # print(policy(torch.zeros(5, 1)))
