#!/bin/sh


ITERS=10000
NUM_TASK=11
SCENARIO=class
DATASET=AZ
MEMORY_BUDGET=100
REPLAY_CONFIG=aws

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START IFS ############'
counter=1
while [ $counter -le 10 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --scenario=${SCENARIO} --replay_config=${REPLAY_CONFIG} --ifs_option=${IFS_OPTION} --memory_budget=${MEMORY_BUDGET}
done
echo All done
