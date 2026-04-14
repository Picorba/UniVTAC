#!/bin/bash
#SBATCH --job-name=univtac
#SBATCH --time=01:00:00
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
module load singularity

unset PYTHONPATH
unset PYTHONHOME
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_EXE
unset CONDA_PYTHON_EXE

export SINGULARITYENV_ACCEPT_EULA=Y
export SINGULARITYENV_PRIVACY_CONSENT=Y
export SINGULARITYENV_HEADLESS=1
export SINGULARITYENV_PREPEND_PATH="/isaac-sim:/isaac-sim/kit/python/bin"
export SINGULARITYENV_WARP_CACHE_PATH=$SCRATCH/isaac/warp_cache
export SINGULARITYENV_MPLCONFIGDIR=$SCRATCH/isaac/matplotlib_cache

# Create dirs
mkdir -p $SCRATCH/isaac/{kit_cache,kit_data,kit_logs,ov_cache,pip_cache,glcache,computecache,logs,data,warp_cache,matplotlib_cache}
mkdir -p $SCRATCH/isaac/ov_cache/texturecache  
mkdir -p $SCRATCH/logs/univtac
mkdir -p $SCRATCH/isaac/home_fake
# Auto-restore patched user.config.json if missing
CONFIG_PATH=$SCRATCH/isaac/kit_data/Kit/Isaac-Sim/4.5/user.config.json
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[INFO] user.config.json missing, restoring patched version..."
    mkdir -p $(dirname $CONFIG_PATH)
    cp $WORK/isaac/user.config.json.patched $CONFIG_PATH
fi

srun singularity exec --nv \
  -B $SCRATCH/isaac/home_fake:/linkhome/rech/genmib01/usf98cb \
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
  -B $SCRATCH/isaacsim_assets:/root/isaac_assets:ro \
  -B /lustre/fsn1/projects/rech/wac/usf98cb/UniVTAC:/workspace/tacex \
  -B $SCRATCH/logs/univtac:/workspace/tacex/logs \
  -B /lustre/fsn1/projects/rech/wac/usf98cb/UniVTAC/data:/workspace/tacex/data \
  $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-jz-cuda126.sif \
  bash -lc '
      export DISPLAY="" && \
      export PYTHONPATH=/workspace/tacex:$PYTHONPATH && \
      export CUDA_LAUNCH_BLOCKING=1 && \
      export TORCH_USE_CUDA_DSA=1 && \
      /isaac-sim/python.sh \
      /workspace/tacex/scripts/collect_data.py \
      lift_bottle demo --start_seed 1 --max_seed 1 --episode_num 1
  '

EXIT_CODE=$?
echo "[INFO] Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE