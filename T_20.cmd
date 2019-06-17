universe = vanilla

executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python

output = T_20.$(Process).out

error = T_20.$(Process).err

log = condor.log

arguments = -u -m cartpole.learn --T=20 --use_pathwise_grad --use_conv_disc --policy_iter=200 --description=\"T=20without-gpu\"

queue 1

