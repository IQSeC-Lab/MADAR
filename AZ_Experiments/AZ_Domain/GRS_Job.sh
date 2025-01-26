#!/bin/sh

GPU_NUMBER=1
SCENARIO=Domain
REPLAY_CONFIG=GRS
GRS_JOINT=True
MEMORY_BUDGET=200000


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'----------' ${SCENARIO} $'----------'
echo $'##### START' ${REPLAY_CONFIG} $'#####'

counter=1
while [ $counter -le 10 ]
do
echo start w/ $counter time 
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python GRS.py --memory_budget=${MEMORY_BUDGET}
echo done w/ $counter time
((counter++))
done
echo All done


