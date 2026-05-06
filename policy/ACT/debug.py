import pickle
import numpy as np

# Load with the newer numpy (in a venv or container where numpy>=1.25 is available)
with open("act_ckpt/act-pick_and_place_fruits/demo-50/train_config/dataset_stats.pkl", "rb") as f:
    stats = pickle.load(f)

# Re-save with your current environment's numpy
with open("sact_ckpt/act-pick_and_place_fruits/demo-50/train_config/dataset_stats.pkl", "wb") as f:
    pickle.dump(stats, f)