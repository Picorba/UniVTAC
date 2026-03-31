#! /bin/bash

TASK_NAME=$1
CONFIG_NAME=${2}
GPU=${3:-0}
START_SEED=${4:--1}
MAX_SEED=${5:--1}
EPISODE=${6:--1}

if [ "$TASK_NAME" = "lift_oscar" ]; then
    for DENSITY in 100 500 2000 5000 10000; do
        echo "=========================================="
        echo "lift_oscar: collecting with density=${DENSITY}"
        echo "=========================================="
        export LIFT_OSCAR_DENSITY=${DENSITY}
        /isaac-sim/python.sh scripts/collect_data.py \
            $TASK_NAME $CONFIG_NAME \
            --start_seed $START_SEED \
            --max_seed $MAX_SEED \
            --episode_num $EPISODE \
            --gpu $GPU \
            --save_subdir density_${DENSITY}
    done
else
    /isaac-sim/python.sh scripts/collect_data.py \
        $TASK_NAME $CONFIG_NAME \
        --start_seed $START_SEED \
        --max_seed $MAX_SEED \
        --episode_num $EPISODE \
        --gpu $GPU
fi
