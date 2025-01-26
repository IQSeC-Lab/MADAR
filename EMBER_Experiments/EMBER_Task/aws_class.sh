#!/bin/sh


NUM_TASK=11
SCENARIO=class
DATASET=ember
GPU_NUMBER=3
NUM_REPLAY_SAMPLE=3000
REPLAY_PORTION=1.0
REPLAY_CONFIG=aws
CNT_RATE=0.1
ITERS=10000

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START IFS ############'
counter=1
while [ $counter -le 1 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --iters=${ITERS} --scenario=${SCENARIO} --replay=offline --replay_portion=${REPLAY_PORTION} --replay_config=${REPLAY_CONFIG} --num_replay_sample=${NUM_REPLAY_SAMPLE} --cnt_rate=${CNT_RATE}
done
echo All done
