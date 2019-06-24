import argparse

# *** ARGUMENT SET UP ***
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", help="Train using a GPU if available", action="store_true")
parser.add_argument("--result_dir_name",
                    help="Name of directory to place results")
parser.add_argument("--policy_lr", type=float,
                    help="Learning rate for the policy, default is 1e-2")
parser.add_argument("--disc_lr", type=float,
                    help="Learning rate for the discriminator, default is 1e-2")
parser.add_argument("--policy_iter", type=int,
                    help="Number of iterations to optimise policy (default is 50)")
parser.add_argument("--description", help="Description the experiment")
parser.add_argument("--T", type=int, help="Number of predicted timesteps")
parser.add_argument("--use_score_func_grad",
                    help="Use score function gradient method for optimising policy", action="store_true")
parser.add_argument("--use_pathwise_grad",
                    help="Use pathwise gradient method for optimising policy", action="store_true")
parser.add_argument("--use_max_log_prob",
                    help="Use maxising the log prob of forward method for optimising policy", action="store_true")
parser.add_argument("--use_state_to_state",
                    help="The descrimator should only work on state to state pairs and not a whole trajectory", action="store_true")
parser.add_argument(
    "--use_conv_disc", help="Set whether to use a Discriminator with a starting convolutional layer", action="store_true")
parser.add_argument("--policy", help="nn | deepnn | rbf | optimal. Default = deepnn")
parser.add_argument("--batch_size", type=int,
                    help="Batch size for policy optimisation (default = Number of training data)")
parser.add_argument("--with_x0", help="If x0 should also be considered when matching the trajectories (default = false)", action="store_true")
parser.add_argument("--fix_seed", help="Fix seed should be set to a default", action="store_true")
parser.add_argument("--num_expr", help="Number of experience/interaction of agent with environment (default == 50)", type=int)

args = parser.parse_args()