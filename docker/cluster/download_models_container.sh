#!/bin/bash
#SBATCH --job-name=cache_torch_weights
#SBATCH --partition=prepost
#SBATCH --account=wac@cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/cache_torch_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/wac/usf98cb/logs/univtac/cache_torch_%j.err

module purge
module load singularity

mkdir -p $SCRATCH/isaac/torch_cache/hub/checkpoints
mkdir -p $SCRATCH/isaac/hf_cache

srun singularity exec \
-B $SCRATCH/isaac/torch_cache:/torch_cache \
-B $SCRATCH/isaac/hf_cache:/hf_cache \
-B $SCRATCH/isaac/home_fake:/root \
-B $SCRATCH/isaac/pip_extra:/pip_extra \
$SINGULARITY_ALLOWED_DIR/isaac-lab-univtac-jz-cuda126.sif \
env HOME=/root \
    TORCH_HOME=/torch_cache \
    HF_HOME=/hf_cache \
    PYTHONPATH=/pip_extra \
/isaac-sim/python.sh -c "
import os
print('TORCH_HOME:', os.environ.get('TORCH_HOME'))
import torchvision.models as m
print('Downloading ResNet18...')
m.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
print('Done.')
from transformers import AutoConfig, AutoProcessor, SiglipModel
print('Downloading SigLIP config...')
AutoConfig.from_pretrained('google/siglip-base-patch16-224')
print('Downloading SigLIP model...')
SiglipModel.from_pretrained('google/siglip-base-patch16-224')
print('Downloading SigLIP processor...')
AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
print('Done.')
"