#!/bin/bash
task_config=${1:-"clean"}
expert_data_num=${2:-100}
seed=${3:-0}
gpu_id=${4:-0}
train_config=${5:-"train_config"}

TASKS=(
    #lift_bottle
    #lift_can
    #insert_hole
    #insert_tube
    #insert_HDMI
    #pull_out_key
    #put_bottle_in_shelf
    grasp_classify
)

for task_name in "${TASKS[@]}"; do
    echo "=== Training: ${task_name} ==="
    sh train.sh "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${gpu_id}"
done
