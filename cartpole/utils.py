import torch
import pandas as pd
import os


def get_expert_data():
    """
    output: N X T x 4
    """
    result = torch.empty(64, 40, 4)

    for i in range(64):
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/expert_data/clockwise/data{i+1}.tsv', delimiter='\t')

        result[i, :, :] = torch.from_numpy(
            df.loc[:, ['x', 'v', 'dtheta', 'theta']].to_numpy())

    return result


def get_training_data(s_a_pairs, traj):
    """
    s_a_pairs: T x D+F
    traj: T+1 x D
    output x: T x D+1+F
    output y: T x D
    Converts a trajectory into the form to be used a target data for the GP,
    i.e into x_t+1 - x_t
    """
    D = traj.shape[-1]

    # replaces theta in the state with sin(theta) and cos(theta) to exploit
    # the wrapping around of angles
    x = torch.cat((convert_to_aux_state(
        s_a_pairs[:, :D], D), s_a_pairs[:, D:]), dim=1)

    y = traj[1:] - traj[:-1]
    return x, y


def convert_to_aux_state(state, D):
    """
    state: (N x) D
    output: N x S
    """
    state = state.view(-1, D)

    return torch.cat((state[:, :D-1], torch.sin(state[:, D-1:D]), torch.cos(state[:, D-1:D])), dim=1)
