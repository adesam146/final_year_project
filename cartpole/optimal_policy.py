import torch
import torch.nn as nn
from rbf_policy import RBFPolicy


class OptimalPolicy(RBFPolicy):
    # Consider making an nn module
    def __init__(self, u_max, device):
        super().__init__(u_max, 5, 10, device)

        self.X = nn.Parameter(torch.tensor([[-4.13779749423284,	0.388195736104642,	0.311289659549716,	-2.16182462811072,	0.987866588280126],
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
                                            [-2.29523332434410,	0.587459048200402,	-0.127238083265639,	-2.12728785878586,	0.887069801692657]], device=self.device))

        self.log_l = nn.Parameter(torch.tensor([2.34165811758010,
                                                0.652757197256950,
                                                1.35740406373520,
                                                0.520681839124039,
                                                -0.309966200635860], device=self.device))

        self.Y = nn.Parameter(torch.tensor([-14.7283720839688,
                                            -19.6035046557288,
                                            -2.35525991310959,
                                            17.8296433086668,
                                            -9.79267204942738,
                                            16.6193178641639,
                                            -11.4750931214345,
                                            6.33739805806107,
                                            -1.51330986923437,
                                            -0.589476409536458], device=self.device))


if __name__ == "__main__":
    print("TESTING OPTIMAL POLICY")
    import numpy as np
    torch.set_default_dtype(torch.float64)

    # Testing Policy, TODO: Convert to test
    policy = OptimalPolicy(10, device=torch.device('cpu'))

    for name, param in policy.named_parameters():
        print("name:", name)
        print("param:", param)

    # Testing against output from optimal policy in matlab
    # with the optimal parameters hardcorded into policy.
    test1 = policy(torch.tensor([0, 0, 0, 0, 1.0]))
    np.testing.assert_almost_equal(
        test1.detach().numpy(), np.array([3.771637760361378]), decimal=4)

    test2 = policy(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(
        test2.detach().numpy(), np.array([-1.417889544030735]), decimal=4)

    test3 = policy(torch.tensor([0.5, 1, 1.5, 2, 2.5]))
    np.testing.assert_almost_equal(
        test3.detach().numpy(), np.array([-3.679293910231314]), decimal=4)

    print("OPTIMAL POLICY TEST SUCCESSFULL")

    # print("Weights", policy.weights)
    # unsquashed = policy.weights[0] * torch.exp(-0.5 * policy.centers[0, 0]**2 * 1/torch.exp(policy.ln_vars)) + policy.weights[1] * torch.exp(-0.5 * policy.centers[0, 1]**2 * 1/torch.exp(policy.ln_vars))
    # print("Unsquashed", unsquashed)
    # print("Squashed", policy.squash(unsquashed))
    # print(policy(torch.zeros(5, 1)))
