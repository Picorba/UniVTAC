#!/usr/bin/env python3
"""
Generate norm_stats.json from already-processed episode HDF5 files.
Reads data incrementally to avoid OOM.
Usage: python gen_norm_stats.py <save_dir> <task_name> <task_config>
"""
import os
import h5py
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", help="Directory with episode_*.hdf5 files")
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    episode_files = sorted(save_dir.glob("episode_*.hdf5"), key=lambda x: int(x.stem.split("_")[1]))
    num_episodes = len(episode_files)
    print(f"Found {num_episodes} episodes in {save_dir}")

    # Welford online algorithm accumulators for qpos/action
    qpos_all, action_all = [], []
    # For tactile images: accumulate channel-wise sums for mean/std
    left_sum = np.zeros(3, dtype=np.float64)
    left_sq_sum = np.zeros(3, dtype=np.float64)
    right_sum = np.zeros(3, dtype=np.float64)
    right_sq_sum = np.zeros(3, dtype=np.float64)
    tac_count = 0

    camera_names = None
    has_wrist = False

    for ep_file in tqdm(episode_files, desc="Computing stats"):
        with h5py.File(ep_file, "r") as f:
            qpos = f["observations/qpos"][()]
            action = f["action"][()]
            left = f["observations/images/cam_left_tactile"][()].astype(np.float32) / 255.0
            right = f["observations/images/cam_right_tactile"][()].astype(np.float32) / 255.0
            if camera_names is None:
                has_wrist = "cam_wrist" in f["observations/images"]
                camera_names = ["cam_high", "cam_wrist"] if has_wrist else ["cam_high"]

            qpos_all.append(qpos)
            action_all.append(action)

            # Accumulate tactile stats channel-wise (pixels × timesteps)
            left_flat = left.reshape(-1, 3)
            right_flat = right.reshape(-1, 3)
            left_sum += left_flat.sum(axis=0)
            left_sq_sum += (left_flat ** 2).sum(axis=0)
            right_sum += right_flat.sum(axis=0)
            right_sq_sum += (right_flat ** 2).sum(axis=0)
            tac_count += left_flat.shape[0]

    qpos_all = np.concatenate(qpos_all, axis=0)
    action_all = np.concatenate(action_all, axis=0)

    left_mean = (left_sum / tac_count).astype(np.float32)
    left_std = np.sqrt(np.maximum(left_sq_sum / tac_count - left_mean ** 2, 0)).astype(np.float32)
    right_mean = (right_sum / tac_count).astype(np.float32)
    right_std = np.sqrt(np.maximum(right_sq_sum / tac_count - right_mean ** 2, 0)).astype(np.float32)

    stats = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "num_episodes": num_episodes,
        "camera": camera_names,
        "tactile": ["cam_left_tactile", "cam_right_tactile"],
        "qpos_mean": qpos_all.mean(axis=0).astype(np.float32).tolist(),
        "qpos_std": qpos_all.std(axis=0).astype(np.float32).tolist(),
        "qpos_min": qpos_all.min(axis=0).astype(np.float32).tolist(),
        "qpos_max": qpos_all.max(axis=0).astype(np.float32).tolist(),
        "action_mean": action_all.mean(axis=0).astype(np.float32).tolist(),
        "action_std": action_all.std(axis=0).astype(np.float32).tolist(),
        "action_min": action_all.min(axis=0).astype(np.float32).tolist(),
        "action_max": action_all.max(axis=0).astype(np.float32).tolist(),
        "left_tac_mean": left_mean.tolist(),
        "left_tac_std": left_std.tolist(),
        "right_tac_mean": right_mean.tolist(),
        "right_tac_std": right_std.tolist(),
    }

    out_path = save_dir / "norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
