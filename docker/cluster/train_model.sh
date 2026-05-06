#!/bin/bash
#SBATCH --job-name=train     
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/%x_%j.err
#SBATCH -A wac@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

module purge
module load anaconda-py3/2023.09
module load cuda/11.8.0

source activate /lustre/fswork/projects/rech/wac/usf98cb/envs/aloha

export HF_HOME=$WORK/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$WORK/.cache/huggingface/hub
export TORCH_HOME=$WORK/.cache/torch
export MPLCONFIGDIR=$WORK/.cache/matplotlib
mkdir -p $HF_HOME $HUGGINGFACE_HUB_CACHE $TORCH_HOME $MPLCONFIGDIR

# ✅ Force offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd $SCRATCH/UniVTAC/policy/ACT
bash train.sh pick_and_place_fruits demo 50 0 0
