#!/bin/bash
#SBATCH --job-name=check_lbfgs_90a
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/check_lbfgs_90a_%j.out
#SBATCH -A wac@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

module purge
module load singularity

echo "=== lbfgs_step_cu arch ==="
singularity exec $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-h100a.sif \
  bash -c "strings /isaac-sim/kit/python/lib/python3.10/site-packages/curobo/curobolib/lbfgs_step_cu.cpython-310-x86_64-linux-gnu.so | grep '^\.target sm_'"