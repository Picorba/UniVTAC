#!/bin/bash
#SBATCH --job-name=install_pip_extra
#SBATCH --partition=prepost
#SBATCH --account=wac@cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/install_pip_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/install_pip_%j.err

module purge
module load singularity

mkdir -p $SCRATCH/isaac/pip_extra

srun singularity exec \
  -B $SCRATCH/isaac/pip_extra:/pip_extra \
  -B $SCRATCH/isaac/home_fake:/root \
  $SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-jz-cuda126.sif \
  env HOME=/root \
  /isaac-sim/kit/python/bin/python3 -m pip install \
  --target=/pip_extra \
  --no-deps \
  sentencepiece \
  protobuf