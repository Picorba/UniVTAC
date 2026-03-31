#!/bin/bash
#SBATCH --job-name=univtac
#SBATCH --time=00:20:00
#SBATCH --partition=visu
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/%x_%j.err
#SBATCH -A wac@v100

cd ${SLURM_SUBMIT_DIR}
module purge
module load singularity
module load cuda
set -x
mkdir -p /var/tmp/singularity

CODE=$SCRATCH/UniVTAC
LOGS=$SCRATCH/logs/univtac
DATA=$CODE/data
mkdir -p $LOGS

srun singularity exec --nv \
--bind $CODE:/workspace/tacex \
--bind $LOGS:/workspace/tacex/logs \
--bind $DATA:/workspace/tacex/data \
--bind /lustre/fswork/projects/rech/wac/usf98cb:/linkhome/rech/genmib01/usf98cb \
--bind /lustre/fswork/projects/rech/wac/usf98cb:/lustre/fswork/projects/rech/wac/usf98cb \
--bind /lustre/fswork/projects/rech/wac/usf98cb/isaac-kit-data:/isaac-sim/kit/data \
--bind /lustre/fswork/projects/rech/wac/usf98cb/isaac-kit-cache:/isaac-sim/kit/cache \
--env OMNI_KIT_ALLOW_ROOT=1 \
--env HEADLESS=1 \
$CONTAINER \
/isaac-sim/python.sh /workspace/tacex/scripts/collect_data.py \
lift_bottle demo \
--start_seed 1 \
--max_seed 1 \
--episode_num 1 \
--gpu 0

echo "[INFO] Job finished with exit code $?"