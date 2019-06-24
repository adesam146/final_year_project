universe = vanilla
executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python
output = experiments/run-19.cmd.$(Process).out
error = experiments/run-19.cmd.$(Process).err
log = condor.log
arguments = -u -m cartpole.learn --T=10 --use_pathwise_grad --use_conv_disc --policy=deepnn --num_expr=20 --policy_iter=100 --policy_lr=0.001 --disc_lr=0.01 --result_dir_name=experiments/run-19.cmd --description=\"experiments/run-19.cmd\"
queue 1