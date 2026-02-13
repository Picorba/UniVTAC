# UniVTAC

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on top of **NVIDIA Isaac Lab** and **TacEx (UIPC-based tactile simulation)**. It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across a diverse suite of contact-rich manipulation tasks — all with high-fidelity tactile feedback from simulated GelSight Mini, GF225, or XenSews sensors.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Data Collection](#data-collection)
- [Policies (Train & Eval)](#policies-train--eval)
  - [Deploy Your Policy](#deploy-your-policy)
  - [ACT](#act)
  - [Ablation](#ablation)
  - [TactileACT](#tactileact)
  - [UniT](#unit)
- [Advanced Usage: Customization](#advanced-usage-customization)
  - [Adding New Tasks](#adding-new-tasks)
- [Task Gallery](#task-gallery)
- [Configuration Reference](#configuration-reference)

---

## Project Structure

```
UniVTAC/
├── assets/                      # Robot URDFs, scene USD files, textures
│   ├── embodiments/             # Robot descriptions (Franka, UR5e, ...)
│   ├── objects/                 # Manipulated object USD assets
│   ├── scene/                   # Scene assets (plates, lights, ...)
│   └── textures/                # Domain randomization textures
├── envs/                        # Task environments
│   ├── _base_task.py            # Base task class (all tasks inherit from this)
│   ├── _global.py               # Global paths and constants
│   ├── collect.py               # Data collection-specific task overrides
│   ├── lift_bottle.py           # Task: Lift a bottle
│   ├── insert_HDMI.py           # Task: Insert an HDMI connector
│   ├── insert_hole.py           # Task: Peg-in-hole insertion
│   ├── ...                      # Other task definitions
│   ├── robot/                   # Robot manager & curobo planner
│   ├── sensors/                 # Camera and tactile sensor managers
│   └── utils/                   # Data I/O, transforms, actor utilities
├── policy/                      # Policy implementations
│   ├── _base_policy.py          # Base policy interface
│   ├── task_settings.json       # Per-task camera/downsample settings
│   ├── ACT/                     # ACT policy
│   ├── Ablation/                # ACT ablation variants
│   ├── UniT/                    # UniT (Diffusion Policy) implementation
│   └── TactileACT/              # TactileACT with CLIP pretraining
├── task_config/                 # Data collection configurations (YAML)
│   ├── demo.yml                 # Default demo config (100 episodes)
│   └── contact.yml              # Contact-focused config (15 episodes)
├── scripts/                     # Entry-point scripts
│   ├── collect_data.py          # Single-process data collection
│   ├── parallel_collect_data.py # Multi-process data collection
│   ├── eval_policy.py           # Single-process policy evaluation
│   ├── parallel_eval_policy.py  # Multi-process policy evaluation
│   └── visualize.py             # Data visualization
├── third_party/
│   └── TacEx/                   # Modified TacEx (build from THIS source)
├── collect_data.sh              # Data collection launcher
├── parallel_collect.sh          # Parallel data collection launcher
├── eval_policy.sh               # Policy evaluation launcher
└── parallel_eval.sh             # Parallel policy evaluation launcher
```

---

## Requirements

| Dependency | Version / Notes |
|---|---|
| Python | 3.10 |
| NVIDIA Isaac Sim / Isaac Lab | Required for physics simulation |
| [TacEx](https://github.com/DH-Ng/TacEx) | **Must be built from the local `third_party/TacEx` source** (contains project-specific modifications) |
| [cuRobo](https://curobo.org) | Required for GPU-accelerated motion planning |
| CUDA-compatible GPU | Required for simulation and policy training |

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/byml-c/UniVTAC.git
cd UniVTAC
```

### Step 2: Create a Conda Environment

```bash
conda create -n UniVTAC python=3.10 -y
conda activate UniVTAC
```

### Step 3: Install TacEx (Modified Source)

> **Important:** Do **not** install TacEx from the public repository. UniVTAC requires a modified version of TacEx that is bundled in `third_party/TacEx`. Some internal APIs have been adapted for UniVTAC's tactile sensor pipeline.

Install TacEx from the local source by following the [TacEx Local Installation Guide](https://github.com/DH-Ng/TacEx/blob/main/docs/source/installation/Local-Installation.md), but point the installation to `./third_party/TacEx` instead of a freshly cloned repo:

```bash
cd third_party/TacEx
# Follow the TacEx local installation instructions from here
# (includes building libuipc and tacex_uipc from source)
cd ../..
```

### Step 4: Install cuRobo

cuRobo is used for GPU-accelerated collision-aware motion planning. Follow the official [cuRobo Installation Guide](https://curobo.org/get_started/1_install_instructions.html).

---

## Data Collection

UniVTAC's data synthesizer enables fully automated data collection by executing scripted manipulation policies (defined in the `envs/` directory) in combination with the cuRobo motion planner. Data collection is configured through task-specific configuration files in `task_config/`, which define parameters such as the target tactile sensor type, observation modalities, texture randomization, and the number of episodes to collect.

The pipeline iterates over random seeds, executing the scripted policy for each seed and saving observation data on success. Failed seeds are skipped automatically, and progress is tracked in `suc_map.txt` to support resuming from interruptions. The entire process is fully automated — just run a single command to get started.

Running the following command will start data collection for the specified task:

```bash
bash collect_data.sh ${task_name} ${config_name} ${gpu_id}
# Example: bash collect_data.sh lift_bottle demo 0
```

For faster collection with multiple parallel simulation workers:

```bash
bash parallel_collect.sh ${task_name} ${config_name} ${gpu_id} [num_processes]
# Example: bash parallel_collect.sh lift_bottle demo 0 3
```

All available `task_name` options correspond to Python modules in the `envs/` directory (e.g., `lift_bottle`, `insert_HDMI`, `pull_out_key`, `grasp_classify`, etc.). The `config_name` parameter specifies a YAML configuration file in `task_config/` (without the `.yml` extension). The `gpu_id` parameter specifies which GPU to use and should be set to an integer in the range `0` to `N-1`, where `N` is the number of GPUs available on your system.

After data collection is completed, the collected data will be stored under `data/${config_name}/${task_name}/`:

- Each episode's observation and action data are saved as an individual HDF5 file in the `hdf5/` directory.
- Visualization videos of each episode (combining camera and tactile views) can be found in the `video/` directory.
- Per-episode metadata (step counts, timing, success/failure results) is stored in `metadata.json`.
- The `suc_map.txt` and `scene/` directory are auxiliary outputs generated during the data collection process.

---

## Policies (Train & Eval)

- [Deploy Your Policy](#deploy-your-policy)
- [ACT](#act)
- [Ablation](#ablation)
- [TactileACT](#tactileact)
- [UniT](#unit)

Each policy is a self-contained module under `policy/` with its own data processing, training, and deployment scripts. All policies share a unified evaluation entry point at the project root:

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
```

For parallel evaluation over many seeds:

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

The evaluation results, including videos and success rate logs, will be saved in the `eval_result/` directory under the project root.

---

### Deploy Your Policy

To deploy and evaluate your own policy in UniVTAC, you need to create three files under `policy/YourPolicy/`:

**1. `deploy_policy.py`** — Implements the policy interface. The following components must be defined:

```python
# policy/YourPolicy/deploy_policy.py
import torch
from policy._base_policy import BasePolicy

class Policy(BasePolicy):
    def __init__(self, args):
        """
        Load your model. `args` is a dict containing all fields from
        deploy.yml, plus runtime fields:
            - args['task_name']   : str — the current task name
            - args['task_config'] : str — the task config file stem
        """
        super().__init__(args)
        self.model = load_your_model(args)

    def encode_obs(self, observation):
        """
        Post-process raw observation into your model's input format.

        The observation dict has the following structure:
            observation = {
                "observation": {
                    "head":  {"rgb": Tensor([H, W, 3])},   # HWC, uint8 [0-255]
                    "wrist": {"rgb": Tensor([H, W, 3])},
                },
                "tactile": {
                    "left_gsmini":  {"rgb_marker": Tensor([H, W, 3]), "depth": ...},
                    "right_gsmini": {"rgb_marker": Tensor([H, W, 3]), "depth": ...},
                },
                "embodiment": {
                    "joint": Tensor([9]),   # 7 arm DOFs + 2 gripper joints
                    "ee":    Tensor([7]),   # position(3) + quaternion(4)
                },
            }
        """
        return your_processed_obs

    def eval(self, task, observation):
        """
        Run one inference step. Called in a loop until the task
        reaches step_lim or reports success.
        """
        obs = self.encode_obs(observation)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')
        # action_type options:
        #   'qpos'     — joint positions: Tensor([8])  (7 arm + 1 gripper)
        #   'ee'       — end-effector pose: Tensor([8]) (position(3) + quaternion(4) + gripper)
        #   'delta_ee' — delta end-effector: Tensor([7]) (delta_position(3) + delta_rotation(3) + delta_gripper)

    def reset(self):
        """Reset internal state (e.g., temporal buffers) at the start of each episode."""
        if hasattr(self.model, 'reset'):
            self.model.reset()
```

**2. `deploy.yml`** — Specifies deployment parameters, which are passed to `Policy.__init__()` as `args`:

```yaml
# policy/YourPolicy/deploy.yml
policy_name: YourPolicy      # Must match the folder name under policy/
seed: 0
ckpt_setting: 50             # Checkpoint identifier (e.g., number of training episodes)
instruction_type: seen
instuction_file: null         # Custom instruction file (null = use task name)
```

**3. (Optional) Update `policy/task_settings.json`** — Register per-task camera settings if needed:

```json
{
    "lift_bottle": {
        "camera_type": "head",
        "downsample": 1
    }
}
```

**Run evaluation:**

```bash
bash eval_policy.sh lift_bottle demo YourPolicy/deploy 0
```

---

### ACT

Action Chunking with Transformers — visuotactile variant with ResNet18 tactile encoders.

**1. Prepare Training Data**

This step converts the collected UniVTAC HDF5 data into the format required for ACT training. The `expert_data_num` parameter specifies the number of trajectory episodes to use.

```bash
cd policy/ACT
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# Example: bash process_data.sh lift_bottle demo 50
```

**2. Train Policy**

This step launches the training process. Training hyperparameters are defined in `train_config.yml` (or a custom config file).

```bash
cd policy/ACT
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id} [train_config]
# Example: bash train.sh lift_bottle demo 50 0 0
```

Checkpoints are saved to `policy/ACT/act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/${train_config}/`.

**3. Eval Policy**

```bash
bash eval_policy.sh ${task_name} ${task_config} ACT/deploy ${gpu_id}
# Example: bash eval_policy.sh lift_bottle demo ACT/deploy 0
```

Alternatively, use the all-in-one script that runs data processing, training, and evaluation sequentially:

```bash
cd policy/ACT
bash one.sh ${task_name} ${task_config} ${gpu_id} [train_config] [expert_data_num]
# Example: bash one.sh lift_bottle demo 0
```

Append `-e` to skip training and jump directly to evaluation.

---

### Ablation

ACT ablation study variants for comparing different observation modalities (vision-only, tactile-only, frozen encoders, etc.). The workflow is identical to ACT, with different training configs selecting different ablation settings.

**1. Prepare Training Data**

```bash
cd policy/Ablation
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# Example: bash process_data.sh lift_bottle demo 50
```

**2. Train Policy**

Use different `train_config` files to run specific ablation experiments (e.g., `train_config_vision`, `train_config_tactile_full`, `train_config_freeze`, etc.):

```bash
cd policy/Ablation
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id} ${train_config}
# Example: bash train.sh lift_bottle demo 50 0 0 train_config_vision
```

**3. Eval Policy**

```bash
bash eval_policy.sh ${task_name} ${task_config} Ablation/deploy ${gpu_id}
# Example: bash eval_policy.sh lift_bottle demo Ablation/deploy 0
```

Or use the all-in-one script:

```bash
cd policy/Ablation
bash one.sh ${task_name} ${task_config} ${gpu_id} ${train_config} [expert_data_num]
# Example: bash one.sh lift_bottle demo 0 train_config_vision
```

---

### TactileACT

TactileACT with CLIP-pretrained tactile and vision encoders. This policy first pretrains a contrastive encoder on tactile-vision pairs, then uses it as the backbone for ACT training.

**1. Prepare Training Data**

```bash
cd policy/TactileACT
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# Example: bash process_data.sh insert_lean demo 50
```

**2. CLIP Pretraining**

Pretrain the contrastive tactile-vision encoder:

```bash
cd policy/TactileACT
export CUDA_VISIBLE_DEVICES=${gpu_id}
python clip_pretraining.py ${task_name} ${task_config} ${expert_data_num}
```

Pretrained encoder checkpoints are saved to `policy/TactileACT/clip_models/`.

**3. Train Policy**

```bash
cd policy/TactileACT
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${gpu_id}
# Example: bash train.sh insert_lean demo 50 0
```

**4. Eval Policy**

```bash
bash eval_policy.sh ${task_name} ${task_config} TactileACT/deploy ${gpu_id}
# Example: bash eval_policy.sh insert_lean demo TactileACT/deploy 0
```

Or use the all-in-one script that runs data processing, CLIP pretraining, training, and evaluation sequentially:

```bash
cd policy/TactileACT
bash one.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash one.sh insert_lean demo 0
```

---

### UniT

Universal Manipulation Interface with Diffusion Policy backbone. UniT uses Hydra for configuration management and supports tactile encoders pretrained on contact-rich data.

**1. Install Dependencies**

UniT requires additional packages (`t3` and `diffusion_policy`). Install them from the policy directory:

```bash
cd policy/UniT
bash install_custom_packages.sh
```

**2. Prepare Training Data**

Convert collected data into Zarr format for Diffusion Policy training:

```bash
cd policy/UniT
python process_data.py ${task_name} ${task_config} ${expert_data_num}
# Example: python process_data.py insert_HDMI demo 50
```

**3. Train Policy**

Training is launched via Hydra. Modify `UniT/config/train_tacarena_policy.yaml` to set the `dataset_path` and other parameters, then run:

```bash
cd policy/UniT
python train.py --config-name=train_tacarena_policy.yaml
```

Checkpoints are saved under `policy/UniT/data/outputs/`.

**4. Eval Policy**

Configure `policy/UniT/deploy.yml` (set `ckpt_path`, `n_exec_steps`, `device`, etc.), then run:

```bash
bash eval_policy.sh ${task_name} ${task_config} UniT/deploy ${gpu_id}
# Example: bash eval_policy.sh insert_HDMI demo UniT/deploy 0
```

---

## Advanced Usage: Customization

### Adding New Tasks

Each task in UniVTAC is defined as a Python module in `envs/` that specifies the scene layout, manipulation objects, scripted expert behavior, and success conditions.

#### Step 1: Create the Task Module

Create a new file `envs/my_new_task.py`:

```python
# envs/my_new_task.py
from ._base_task import *
import numpy as np


@configclass
class TaskCfg(BaseTaskCfg):
    """Task-specific configuration overrides."""
    step_lim = 500                          # Max policy steps during evaluation
    adaptive_grasp_depth_threshold = 27.0   # Tactile depth threshold (mm)

    # Override cameras if needed (otherwise inherits BaseTaskCfg defaults)
    # cameras = [...]


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode='collect', render_mode=None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        """
        Called once during scene setup.
        Add all manipulated objects to the scene using the ActorManager.
        """
        obj_pose = Pose([0.7, 0.0, 0.01], [1, 0, 0, 0])
        self.my_object = self._actor_manager.add_from_usd_file(
            name='my_object',
            asset_path="MyObject.usd",     # Relative to assets/objects/
            pose=obj_pose
        )

    def _reset_actors(self):
        """
        Called on each episode reset.
        Randomize object poses for domain randomization.
        """
        noise = self.create_noise(vec=[0.01, 0.02, 0.0], euler=[0, 0, np.pi/18])
        new_pose = Pose([0.7, 0.0, 0.01]).add_offset(noise)
        self.my_object.set_pose(new_pose)

    def pre_move(self):
        """
        Scripted pre-manipulation phase (e.g., approach and grasp).
        Executed before _play_once() during data collection.
        """
        self.delay(10)
        target_pose = self.my_object.get_pose().add_bias([-0.1, 0, 0])
        grasp_pose = construct_grasp_pose(target_pose.p, [0, 0, 1], [1, 0, 0])
        grasp_id = self.my_object.register_point(grasp_pose, type='contact')
        self.move(self.atom.grasp_actor(self.my_object, contact_point_id=grasp_id))

    def _play_once(self):
        """
        Scripted expert behavior for data collection.
        Define the manipulation trajectory here.
        """
        self.move(self.atom.close_gripper())
        # ... define your manipulation sequence ...

    def check_success(self):
        """
        Return True if the task is completed successfully.
        Called during both data collection and policy evaluation.
        """
        # Example: check if object reached target height
        obj_pos = self.my_object.get_pose().p
        return obj_pos[2] > 0.10  # z > 10cm
```

#### Step 2: Add Object Assets

Place your object USD files in `assets/objects/`:

```
assets/objects/MyObject.usd
```

#### Step 3: Add a Task Configuration (Optional)

If the task requires custom collection settings, add a config in `task_config/`:

```yaml
# task_config/my_config.yml
save_dir: ./data

decimation: 1
save_frequency: 2
video_frequency: 2
render_frequency: 0       # Set to 1 to enable GUI rendering

random_texture: false
use_seed: true
episode_num: 100

sensor_type: gsmini       # Options: gsmini, gf225, xensews

observations:
  camera: ['rgb']
  tactile: ['rgb', 'rgb_marker', 'marker', 'depth', 'pose']
  embodiment: ['joint', 'ee']
  actor: true
```

#### Step 4: Register Task in Policy Settings

Add your task to `policy/task_settings.json` so policies know which cameras to use:

```json
{
    "my_new_task": {
        "camera_type": "head",
        "downsample": 1
    }
}
```

#### Step 5: Collect Data and Evaluate

```bash
# Collect expert demonstrations
bash collect_data.sh my_new_task demo 0

# Evaluate an existing policy on the new task
bash eval_policy.sh my_new_task demo ACT/deploy 0
```

---

## Task Gallery

UniVTAC currently includes the following manipulation tasks, all featuring tactile sensing:

| Task | Module | Description |
|---|---|---|
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle off a surface near a wall |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |

---

## Configuration Reference

### Task Configuration (`task_config/*.yml`)

| Field | Type | Default | Description |
|---|---|---|---|
| `save_dir` | `str` | `./data` | Root directory for saving collected data. |
| `decimation` | `int` | `1` | Physics sub-stepping factor. |
| `save_frequency` | `int` | `2` | Save observations every N simulation steps. |
| `video_frequency` | `int` | `2` | Record video frames every N steps. |
| `render_frequency` | `int` | `0` | Render GUI every N steps (`0` = headless). |
| `random_texture` | `bool` | `false` | Enable random texture domain randomization. |
| `use_seed` | `bool` | `true` | Use deterministic seeding for reproducibility. |
| `episode_num` | `int` | `100` | Number of successful episodes to collect. |
| `sensor_type` | `str` | `gsmini` | Tactile sensor type: `gsmini`, `gf225`, or `xensews`. |
| `observations` | `dict` | — | Which observation modalities to record (see below). |

#### Observation Modalities

```yaml
observations:
  camera: ['rgb']                                          # Camera: rgb, depth
  tactile: ['rgb', 'rgb_marker', 'marker', 'depth', 'pose']  # Tactile modalities
  embodiment: ['joint', 'ee']                              # Robot state
  actor: true                                              # Object actor states
```

### Policy Deployment Configuration (`policy/*/deploy.yml`)

| Field | Type | Description |
|---|---|---|
| `policy_name` | `str` | Policy module name (must match a directory under `policy/`). |
| `seed` | `int` | Seed offset for evaluation. |
| `ckpt_setting` | `int` | Checkpoint identifier (e.g., number of training demos). |
| `instruction_type` | `str` | `"seen"` or `"unseen"` — language instruction split. |
| `instuction_file` | `str\|null` | Custom instruction JSON file (default: task name). |

