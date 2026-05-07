#!/bin/bash

BASE_SCRIPT=$SCRATCH/UniVTAC/docker/cluster/collect_data.sh
STEP=20
N_JOBS=5
TASK_NAME=${1:-"pick_and_place_fruits"}
TASK_MODE=${2:-"demo"}             

echo "[INFO] Launching $N_JOBS jobs for task: '$TASK_NAME' mode: '$TASK_MODE'"

for ((i=0; i<N_JOBS; i++)); do
    START=$((i * STEP))
    END=$((START + STEP - 1))
    sbatch --export=ALL,START_SEED=$START,MAX_SEED=$END,TASK_NAME=$TASK_NAME,TASK_MODE=$TASK_MODE \
           --job-name="${TASK_NAME}" \
           $BASE_SCRIPT
done