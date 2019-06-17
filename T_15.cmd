universe = vanilla

executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python

output = T_15.$(Process).out

error = T_15.$(Process).err

log = condor.log

arguments = -u -m cartpole.learn --T=15 --use_pathwise_grad --use_conv_disc --policy_iter=200 --description=\"T=15without-gpu\"

queue 1

