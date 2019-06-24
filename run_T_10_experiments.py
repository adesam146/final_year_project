import argparse
import subprocess
import os

# Created cmd files to be used by Condor

if not os.path.isdir('experiments'):
    os.makedirs('experiments')

# *** ARGUMENT SET UP ***
parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, help="Number of predicted timesteps (default=10)")
parser.add_argument("--runs", type=int, help="Number of independent runs (default=20)")
args = parser.parse_args()

T = args.T or 10
runs = args.runs or 20

policy_lr = 1e-3
disc_lr = 1e-2
policy_iter = 100

filenames = []
for run in range(runs):
    filename = f'experiments/run-{run}.cmd'
    with open(filename, 'w') as f:
        f.write(f'universe = vanilla\n'
                f'executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python\n'
                f'output = {filename}.$(Process).out\n'
                f'error = {filename}.$(Process).err\n'
                'log = condor.log\n'
                f'arguments = -u -m cartpole.learn --T={T} --use_pathwise_grad --use_conv_disc --policy=deepnn --num_expr={20} --policy_iter={policy_iter} --policy_lr={policy_lr} --disc_lr={disc_lr} --result_dir_name={filename} --description=\\"{filename}\\"\n'
                'queue 1')
    filenames.append(filename)

for filename in filenames:
    subprocess.run(["condor_submit", filename])
