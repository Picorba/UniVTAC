# Installation Guide

## Requirements

- System: Linux with NVIDIA GPU
- Python 3.10
- NVIDIA Isaac Sim 4.5 + Isaac Lab 2.1.1
- [NVIDIA cuRobo](https://curobo.org)
- [TacEx](https://github.com/DH-Ng/TacEx): **Must be built from the local `third_party/TacEx` source** (contains project-specific modifications)

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

Install TacEx from the local source by following the [TacEx Local Installation Guide](./third_party\TacEx\docs\source\installation\Local-Installation.md), but point the installation to `./third_party/TacEx` instead of a freshly cloned repo:

```bash
cd third_party/TacEx
# Follow the TacEx local installation instructions from here
# (includes building libuipc and tacex_uipc from source)
cd ../..
```

### Step 4: Install cuRobo

cuRobo is used for GPU-accelerated collision-aware motion planning. Follow the official [cuRobo Installation Guide](https://curobo.org/get_started/1_install_instructions.html).