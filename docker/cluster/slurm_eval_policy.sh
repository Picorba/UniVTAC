#!/bin/bash
#SBATCH --job-name=eval_policy
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

TASK_NAME=${TASK_NAME:-"pick_and_place_fruits"}
TASK_MODE=${TASK_MODE:-"demo"}

module purge
module load singularity
unset PYTHONPATH PYTHONHOME CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE

export SINGULARITYENV_ACCEPT_EULA=Y
export SINGULARITYENV_PRIVACY_CONSENT=Y
export SINGULARITYENV_HEADLESS=1
export SINGULARITYENV_PREPEND_PATH="/isaac-sim:/isaac-sim/kit/python/bin"
export SINGULARITYENV_WARP_CACHE_PATH=$SCRATCH/isaac/warp_cache
export SINGULARITYENV_MPLCONFIGDIR=$SCRATCH/isaac/matplotlib_cache
export SINGULARITYENV_TORCH_HOME=$SCRATCH/isaac/torch_cache
export SINGULARITYENV_HF_HOME=$SCRATCH/isaac/hf_cache            
export SINGULARITYENV_TRANSFORMERS_CACHE=$SCRATCH/isaac/hf_cache/hub 
export SINGULARITYENV_HF_DATASETS_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1

mkdir -p $SCRATCH/isaac/{kit_cache,kit_data,kit_logs,ov_cache,pip_cache,glcache,computecache,logs,data,warp_cache,matplotlib_cache}
mkdir -p $SCRATCH/isaac/ov_cache/texturecache
mkdir -p $SCRATCH/isaac/hf_cache                                  
mkdir -p $SCRATCH/isaac/pip_extra                                   
mkdir -p $SCRATCH/logs/univtac
mkdir -p $SCRATCH/isaac/home_fake

CONFIG_PATH=$SCRATCH/isaac/kit_data/Kit/Isaac-Sim/4.5/user.config.json
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[INFO] user.config.json missing, restoring patched version..."
    mkdir -p $(dirname $CONFIG_PATH)
    cp $WORK/isaac/user.config.json.patched $CONFIG_PATH
fi

echo "[INFO] Running task='$TASK_NAME' mode='$TASK_MODE' seeds=[$START_SEED..$MAX_SEED]"

srun singularity exec --nv \
-B $SCRATCH/isaac/home_fake:/linkhome/rech/genmib01/usf98cb \
-B $SCRATCH/isaac/kit_cache:/isaac-sim/kit/cache \
-B $SCRATCH/isaac/kit_data:/isaac-sim/kit/data \
-B $SCRATCH/isaac/kit_logs:/isaac-sim/kit/logs \
-B $SCRATCH/isaac/ov_cache:/root/.cache/ov \
-B $SCRATCH/isaac/pip_cache:/root/.cache/pip \
-B $SCRATCH/isaac/torch_cache:/torch_cache \
-B $SCRATCH/isaac/glcache:/root/.cache/nvidia/GLCache \
-B $SCRATCH/isaac/computecache:/root/.nv/ComputeCache \
-B $SCRATCH/isaac/logs:/root/.nvidia-omniverse/logs \
-B $SCRATCH/isaac/data:/root/.local/share/ov/data \
-B $SCRATCH/isaac/warp_cache:/root/.cache/warp \
-B $SCRATCH/isaac/hf_cache:/hf_cache \
-B $SCRATCH/isaac/pip_extra:/pip_extra \
-B $SCRATCH/isaacsim_assets:/root/isaac_assets:ro \
-B /lustre/fsn1/projects/rech/wac/usf98cb/UniVTAC:/workspace/tacex \
-B $SCRATCH/logs/univtac:/workspace/tacex/logs \
-B /lustre/fsn1/projects/rech/wac/usf98cb/UniVTAC/data:/workspace/tacex/data \
$SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-jz-cuda126.sif \
bash -lc "
    export DISPLAY='' && \
    export PYTHONPATH=/pip_extra:/workspace/tacex:\$PYTHONPATH && \  # ← CHANGED
    export CUDA_LAUNCH_BLOCKING=1 && \
    export TORCH_USE_CUDA_DSA=1 && \
    /isaac-sim/python.sh \
    /workspace/tacex/scripts/eval_failure_policy.py \
    ${TASK_NAME} ${TASK_MODE} ${MODEL_CONFIG} \
    --start_seed ${START_SEED} --max_seed ${MAX_SEED}
"

EXIT_CODE=$?
echo "[INFO] Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE