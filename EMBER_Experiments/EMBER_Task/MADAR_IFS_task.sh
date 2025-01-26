#!/bin/bash

ITERS=10000
NUM_TASK=11
SCENARIO=task
DATASET=ember
MEMORY_BUDGET=100
REPLAY_CONFIG=ifs


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'##### START' "$1" "$2" "$3" $'#####'

for counter in {1..10}
do
    echo Start w/ $counter time 
    python main.py --metrics --scenario=${SCENARIO} --replay_config=${REPLAY_CONFIG} --ifs_option=${IFS_OPTION} --memory_budget=${MEMORY_BUDGET}
    echo done w/ $counter time
    
done
echo All done
conda deactivate
EOT