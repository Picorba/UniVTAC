#!/bin/bash
#SBATCH --job-name=check_lbfgs
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/check_lbfgs_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/check_lbfgs_%j.err
#SBATCH -A wac@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

module purge
module load singularity

echo "=== Checking ALL curobo kernels arch ==="
singularity exec $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-h100.sif \
  bash -c "find /isaac-sim/kit/python/lib/python3.10/site-packages/curobo -name '*.so' | \
  while read f; do
    result=\$(strings \$f | grep -E '^\.target sm_[0-9]+' | sort -u)
    echo \"--- \$(basename \$f) ---\"
    echo \"\${result:-NO CUDA KERNELS}\"
  done"