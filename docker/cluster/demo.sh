#!/bin/bash
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/%x_%j.err
#SBATCH --job-name=demo
#SBATCH --time=02:00:00
#SBATCH -A wac@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t4


module purge
module load singularity

unset PYTHONPATH
unset PYTHONHOME
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_EXE
unset CONDA_PYTHON_EXE

export SINGULARITYENV_ACCEPT_EULA=Y # required by Isaac
export SINGULARITYENV_PRIVACY_CONSENT=Y # required by Isaac
export SINGULARITYENV_HEADLESS=1
export SINGULARITYENV_WARP_CACHE_PATH=$SCRATCH/isaac/warp_cache

singularity exec --nv \
-B $SCRATCH/isaac/kit_cache:/isaac-sim/kit/cache \
-B $SCRATCH/isaac/kit_data:/isaac-sim/kit/data \
-B $SCRATCH/isaac/kit_logs:/isaac-sim/kit/logs \
-B $SCRATCH/isaac/ov_cache:/root/.cache/ov \
-B $SCRATCH/isaac/pip_cache:/root/.cache/pip \
-B $SCRATCH/isaac/glcache:/root/.cache/nvidia/GLCache \
-B $SCRATCH/isaac/computecache:/root/.nv/ComputeCache \
-B $SCRATCH/isaac/logs:/root/.nvidia-omniverse/logs \
-B $SCRATCH/isaac/data:/root/.local/share/ov/data \
-B $SCRATCH/isaac/warp_cache:/root/.cache/warp \
-B $WORK/isaac/outputs:/root/outputs \
$SINGULARITY_ALLOWED_DIR/isaac-lab-univtac.sif \
bash -lc 'cd /root/outputs && mkdir -p logs && /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/tutorials/00_sim/log_time.py --headless'