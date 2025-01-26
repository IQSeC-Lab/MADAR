#!/bin/sh

ITERS=10000
NUM_TASK=11
SCENARIO=class
DATASET=ember
MEMORY_BUDGET=100
REPLAY_CONFIG=grs


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START GRS ############'
counter=1
while [ $counter -le 10 ]
do
echo Start w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --scenario=${SCENARIO} --replay_config=${REPLAY_CONFIG} --memory_budget=${MEMORY_BUDGET}
echo done w/ $counter time
((counter++))
done
echo All done
