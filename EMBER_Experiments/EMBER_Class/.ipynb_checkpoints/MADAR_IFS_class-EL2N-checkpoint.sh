#!/bin/sh


ITERS=10000
NUM_TASK=11
SCENARIO=class
DATASET=ember
MEMORY_BUDGET=1000
REPLAY_CONFIG=ifs
IFS_OPTION=ratio
GPU_NUMBER=0

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START IFS ############'
counter=1
while [ $counter -le 2 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main-EL2N.py --metrics --scenario=${SCENARIO} --replay_config=${REPLAY_CONFIG} --ifs_option=${IFS_OPTION} --memory_budget=${MEMORY_BUDGET}
done
echo All done
