#!/bin/bash
module purge

module load anaconda-py3/2023.09
module load cuda/11.8.0

ENV_ROOT=/lustre/fswork/projects/rech/wac/usf98cb/envs/aloha
export PATH=$ENV_ROOT/bin:$PATH

export HF_HOME=$WORK/.cache/huggingface
export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$WORK/.cache/huggingface/hub
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

python3 - <<'EOF'
from transformers import (
    AutoConfig,
    AutoProcessor,
    SiglipTokenizer,
    SiglipModel,
)

MODEL_ID = "google/siglip-base-patch16-224"
print(f"Downloading {MODEL_ID}...")

print("  [1/4] AutoConfig...")
AutoConfig.from_pretrained(MODEL_ID)

print("  [2/4] SiglipTokenizer...")
SiglipTokenizer.from_pretrained(MODEL_ID)

print("  [3/4] SiglipModel (full weights)...")
SiglipModel.from_pretrained(MODEL_ID)

print("  [4/4] AutoProcessor...")
AutoProcessor.from_pretrained(MODEL_ID)

print("✅ All components cached successfully.")
print(f"Cache location: $WORK/.cache/huggingface/hub")
EOF