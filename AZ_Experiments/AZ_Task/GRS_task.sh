#!/bin/sh


NUM_TASK=11
SCENARIO=task
DATASET=AZ
GPU_NUMBER=1
MEMORY_BUDGET=500
REPLAY_CONFIG=grs

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START GRS ############'
counter=1
while [ $counter -le 1 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --scenario=${SCENARIO} --replay_config=${REPLAY_CONFIG} --memory_budget=${MEMORY_BUDGET}
done
echo All done
