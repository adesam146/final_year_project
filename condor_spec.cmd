universe = vanilla

executable = /vol/bitbucket/aso115/anaconda3/envs/myvenv/bin/python

output = condor.$(Process).out

error = condor.$(Process).err

log = condor.log

arguments = -u -m cartpole.learn --T=40 --use_pathwise_grad --use_conv_disc --policy_iter=100 --batch_size=16 --description=\"without-gpu\"

queue 1

