import torch
import pandas as pd
import os


def get_expert_data():
    result = torch.empty(59, 40, 4)

    for i in range(59):
      df = pd.read_csv(f'{os.path.dirname(__file__)}/expert_data/clockwise/data{i+1}.tsv', delimiter='\t')

      result[i, :, :] = torch.from_numpy(df.loc[:, ['x', 'v', 'dtheta', 'theta']].to_numpy())

    return result


def get_train_y(traj):
    """
    traj: T+1 x D
    output y: T x D
    Converts a trajectory into the form to be used a target data for the GP,
    i.e into x_t+1 - x_t
    """
    y = traj[1:] - traj[:-1]
    return y
