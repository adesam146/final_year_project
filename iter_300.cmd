universe = vanilla

executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python

output = iter_300.$(Process).out

error = iter_300.$(Process).err

log = condor.log

arguments = -u -m cartpole.learn --T=10 --use_pathwise_grad --use_conv_disc --policy_iter=300 --description=\"T=10without-gpu\"

queue 1

