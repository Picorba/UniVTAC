# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on **NVIDIA Isaac Lab** and **TacEx** (UIPC-based FEM tactile simulation). It provides a pipeline for: collecting scripted expert demonstrations → training visuotactile policies → evaluating them across contact-rich manipulation tasks with GelSight Mini, ViTai GF225, or XenseWS tactile sensors.

## Key Commands

### Data Collection
```bash
# Collect demonstrations for a task (single process)
bash collect_data.sh ${task_name} ${config_name} ${gpu_id}
# Example: bash collect_data.sh lift_bottle demo 0

# Parallel collection (launches multiple Isaac Sim Apps)
bash parallel_collect.sh ${task_name} ${config_name} ${gpu_id} [num_processes]
# Example: bash parallel_collect.sh lift_bottle demo 0 3
```

### Policy Evaluation
```bash
# Evaluate a trained policy
bash eval_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
# Example: bash eval_policy.sh lift_bottle demo ACT/deploy 0

# Parallel evaluation over many seeds
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

### Policy Training (ACT/Ablation/TactileACT)
```bash
# Process collected data
bash policy/ACT/process_data.sh

# Train a policy
bash policy/ACT/train.sh

# Evaluate a single episode
bash policy/ACT/one.sh
```

## Architecture

### Task System (`envs/`)

Tasks follow a two-layer class hierarchy:

1. **`BaseTaskCfg` / `BaseTask`** (`envs/_base_task.py`) — The foundation. `BaseTask` extends `UipcRLEnv` (TacEx's environment class). All configuration is done via `@configclass`-decorated `BaseTaskCfg`.

2. **`TaskCfg` / `Task`** in each task file (e.g., `envs/lift_bottle.py`) — Extends the base. Each task overrides:
   - `create_actors()` — spawns USD objects into the scene via `ActorManager`
   - `_reset_actors()` — randomizes actor poses each episode
   - `pre_move()` — pre-grasp motion executed before data collection begins
   - `_play_once()` — the scripted policy logic (called during collection)
   - `check_success()` — success criterion for the episode

The `mode` parameter (`'collect'` or `'eval'`) controls whether the env saves data or runs a learned policy.

### Core Managers

- **`RobotManager`** (`envs/robot/robot.py`) — Wraps a Franka Panda articulation; handles arm/gripper joint control, IK, and pose queries. Uses **cuRobo** for motion planning via `CuroboPlanner`.
- **`ActorManager`** (`envs/utils/actor.py`) — Manages UIPC deformable/rigid objects (actors). Each `Actor` wraps a `UipcObject` and exposes typed point sets (contact, functional, target, orientation) for grasp planning.
- **`CameraManager`** (`envs/sensors/camera.py`) — Manages RGB/depth cameras (head + wrist views at 480×270).
- **`TactileManager`** (`envs/sensors/tactile.py`) — Manages left/right tactile sensors. Sensor type is configured per task via `cfg.tactile_sensor_type` (`'gsmini'`, `'gf225'`, `'xensews'`).

### Motion Planning Primitives (`envs/utils/atom.py`)

`Atom` is a helper class used inside task scripts to build action sequences:
- `atom.grasp_actor(actor, ...)` — approach + grasp
- `atom.place_actor(actor, target_pose, ...)` — place at target
- `atom.move_by_displacement(x, y, z, ...)` — relative EE move
- `atom.close_gripper()` / `atom.open_gripper()` — gripper control

Actions are `Action` objects passed to `task.move([...])`, which calls `RobotManager.plan_arm()` and executes the trajectory step-by-step.

### `Pose` class (`envs/utils/transforms.py`)

The universal pose representation. Stores `p` (position, `np.ndarray[3]`) and `q` (quaternion wxyz, `np.ndarray[4]`). Key methods:
- `pose.add_bias([x, y, z], coord='world'/'local'/Pose)` — translate
- `pose.add_offset(noise_pose)` — apply noise
- `pose.add_rotation(euler, coord=...)` — rotate
- `pose.rebase(reference_pose)` — express pose in reference frame
- `pose.to_transformation_matrix()` — 4×4 matrix

### Data Pipeline

- During collection, each step's observation dict is saved as `.pkl` in a temp cache, then consolidated into HDF5 via `HDF5Handler.pkls_to_hdf5()`.
- Output structure: `data/${config_name}/${task_name}/hdf5/*.hdf5`, `video/*.mp4`, `metadata.json`, `suc_map.txt`.
- `suc_map.txt` tracks seed success/failure for resumable collection.

### Task Configuration (`task_config/*.yml`)

YAML files control collection parameters: `sensor_type`, `episode_num`, `save_frequency`, `video_frequency`, `render_frequency` (0=headless), `random_texture`, and the `observations` dict specifying which modalities to record.

### Policy System (`policy/`)

Each policy is a self-contained directory with:
- `deploy_policy.py` — defines `Policy(BasePolicy)` with `__init__`, `encode_obs`, `eval`, and `reset`
- `deploy.yml` — deployment config passed as `args` to `Policy.__init__`
- `process_data.py` / `train.py` — data processing and training scripts

`task.take_action(action, action_type)` is the interface between policy and env:
- `'qpos'`: `Tensor([8])` — 7 arm DOFs + 1 gripper
- `'ee'`: `Tensor([8])` — position (3) + quaternion (4) + gripper (1)
- `'delta_ee'`: `Tensor([7])` — delta position (3) + delta rotation (3) + delta gripper (1)

### Asset Paths (`envs/_global.py`)

```python
ASSETS_ROOT         # assets/
OBJECTS_ROOT        # assets/objects/   (USD object files)
EMBODIMENTS_ROOT    # assets/embodiments/  (robot URDFs, cuRobo configs)
SCENE_ASSETS_ROOT   # assets/scene/
TEXTURES_ROOT       # assets/textures/
```

## Adding a New Task

1. Create `envs/my_task.py` inheriting from `BaseTask`/`BaseTaskCfg`
2. Implement `create_actors()`, `_reset_actors()`, `pre_move()`, `_play_once()`, `check_success()`
3. Place USD assets in `assets/objects/`
4. Run: `bash collect_data.sh my_task demo 0`

## Important Notes

- The simulation runs at 120 Hz (`dt=1/120`). `decimation=1` means physics and control run at the same rate.
- `render_frequency=0` enables headless mode (uses livestream); set to 1 to open the Isaac Sim GUI.
- The `PRISM_NAME` environment variable (default `'Default'`) is used as a subdirectory in the save path, useful for organizing multi-machine collections.
- Only Franka Panda is currently supported as the robot arm.
- `adaptive_grasp_depth_threshold` (mm) controls when the gripper stops closing; it is sensor- and task-specific and should be tuned per task.
