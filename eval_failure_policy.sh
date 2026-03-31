#!/bin/bash
TASK_NAME=${1}
TASK_CONFIG=${2}
POLICY_CONFIG=${3}
GPU=${4}

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/eval_failure_policy.py $TASK_NAME $TASK_CONFIG $POLICY_CONFIG "${@:5}"
