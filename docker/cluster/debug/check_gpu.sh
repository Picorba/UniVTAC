#!/bin/bash
#SBATCH --job-name=check_gpu
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/check_gpu_%j.out
#SBATCH -A wac@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

module purge
module load singularity

# Run directly on the node, no container
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvidia-smi -a | grep -E "Product Name|Compute Mode|CUDA Capability"