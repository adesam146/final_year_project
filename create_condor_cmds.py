import argparse
import subprocess
import os

if not os.path.isdir('experiments'):
    os.makedirs('experiments')

# *** ARGUMENT SET UP ***
parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, help="Number of predicted timesteps")
args = parser.parse_args()

T = args.T

policy_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
disc_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
policy_iters = [50, 100, 200, 400]

filenames = []
for policy_iter in policy_iters:
    for policy_lr in policy_lrs:
        for disc_lr in disc_lrs:
            filename = f'experiments/T_{T}-piter_{policy_iter}-plr_{policy_lr}-dlr_{disc_lr}.cmd'
            with open(filename, 'w') as f:
                f.write(f'universe = vanilla\n'
                        f'executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python\n'
                        f'output = T_{T}.$(Process).out\n'
                        f'error = T_{T}.$(Process).err\n'
                        'log = condor.log\n'
                        f'arguments = -u -m cartpole.learn --T={T} --use_pathwise_grad --use_conv_disc --policy_iter={policy_iter} --policy_lr={policy_lr} --disc_lr={disc_lr} --description=\"{filename}\"\n'
                        'queue 1')
            filenames.append(filename)

for filename in filenames:
    subprocess.run(["condor_submit", filename])
