#!/bin/bash
#SBATCH --job-name=univtac
#SBATCH --time=00:20:00
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

export SINGULARITYENV_ACCEPT_EULA=Y
export SINGULARITYENV_PRIVACY_CONSENT=Y
export SINGULARITYENV_HEADLESS=1
export SINGULARITYENV_OMNI_KIT_ALLOW_ROOT=1
export SINGULARITYENV_PREPEND_PATH="/isaac-sim:/isaac-sim/kit/python/bin"
export SINGULARITYENV_WARP_CACHE_PATH=$SCRATCH/isaac/warp_cache
export SINGULARITYENV_MPLCONFIGDIR=$SCRATCH/isaac/matplotlib_cache

# Paths
CODE=$SCRATCH/UniVTAC
LOGS=$SCRATCH/logs/univtac
DATA=$CODE/data

# Create dirs
mkdir -p $SCRATCH/isaac/{kit_cache,kit_data,kit_logs,ov_cache,pip_cache,glcache,computecache,logs,data,warp_cache,matplotlib_cache}
mkdir -p $LOGS

# Auto-restore patched user.config.json if missing
CONFIG_PATH=$SCRATCH/isaac/kit_data/Kit/Isaac-Sim/4.5/user.config.json
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[INFO] user.config.json missing, restoring patched version..."
    mkdir -p $(dirname $CONFIG_PATH)
    cp $WORK/isaac/user.config.json.patched $CONFIG_PATH
fi

srun singularity exec --nv \
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
  $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac.sif \
  bash -lc 'apt-get install -y xvfb -qq 2>/dev/null && Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset & sleep 3 && export DISPLAY=:99 && export PYTHONPATH=/workspace/tacex:$PYTHONPATH && /workspace/isaaclab/isaaclab.sh -p /workspace/tacex/scripts/collect_data.py lift_bottle demo --start_seed 1 --max_seed 1 --episode_num 1 --gpu 0'
EXIT_CODE=$?
echo "[INFO] Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE