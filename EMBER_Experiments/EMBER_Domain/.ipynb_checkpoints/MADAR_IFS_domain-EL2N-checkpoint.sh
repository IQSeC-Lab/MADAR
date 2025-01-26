#!/bin/sh

GPU_NUMBER=1
SCENARIO=Domain
REPLAY_CONFIG=IFS
IFS_OPTION=ratio
MEMORY_BUDGET=400000

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'----------' ${SCENARIO} $'----------'
echo $'##### START' ${REPLAY_CONFIG} ${NUM_REPLAY_SAMPLE} $'#####'

counter=1
while [ $counter -le 2 ]
do
echo start w/ $counter time 
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python MADAR_IFS-EL2N.py --ifs_option=${IFS_OPTION} --memory_budget=${MEMORY_BUDGET}
echo done w/ $counter time
((counter++))
done
echo All done