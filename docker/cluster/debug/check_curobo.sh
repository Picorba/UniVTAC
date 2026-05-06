#!/bin/bash
#SBATCH --job-name=check_physx
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/check_physx_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/check_physx_%j.err
#SBATCH -A wac@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

module purge
module load singularity

echo "=== Checking omni.physx.fabric arch ==="
singularity exec $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-h100.sif \
  bash -c "find /isaac-sim -name '*physx*fabric*' -o -name '*PhysX*' 2>/dev/null | \
  grep '\.so' | head -5 | xargs -I{} sh -c 'echo {} && strings {} | grep -E \"sm_[0-9]+\" | sort -u'"

echo "=== Checking all isaac-sim .so for sm arch ==="
singularity exec $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-h100.sif \
  bash -c "find /isaac-sim/exts -name '*.so' | xargs -I{} sh -c \
  'result=\$(strings {} | grep -E \"sm_[0-9]+\" | sort -u); [ -n \"\$result\" ] && echo \"--- {} ---\" && echo \"\$result\"' \
  2>/dev/null"

echo "=== Check curobo version ==="
# Check cuRobo version
singularity exec $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-h100a.sif \
  bash -c "/isaac-sim/python.sh -c 'import curobo; print(curobo.__version__)'"