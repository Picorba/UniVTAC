# TacArena

# Requirements
- Python 3.10
- TacEx (need to built from source in ./third_party/TacEx, some codes has been modified to fit TacArena)
- curobo (need for motion planning)

# Installation
1. Clone this repository
```
git clone https://github.com/byml-c/TacArena.git
cd TacArena
```

2. Create a virtual environment and activate it
```
conda create -n tacarena python=3.10 -y
conda activate tacarena
```

3. Install TacEx use the modified source code in `third_party/TacEx`, follow the instruction [here](https://github.com/DH-Ng/TacEx/blob/main/docs/source/installation/Local-Installation.md).

4. Install curobo, follow the instruction [here](https://curobo.org/get_started/1_install_instructions.html).

# Collect Data
1. Collection configurations are in `task_config`
2. Run `collect_data.sh` to start data collection, such as
```bash
bash collect_data.sh <task_name> <config_name> <gpu_id>
# e.g. bash collect_data.sh lift_bottle demo 0
```

# Train and Eval Policy
1. Policies are in `policy`.
2. Run `eval_policy.sh` to start evaluation, such as
```bash
bash eval_policy.sh <task_name> <config_name> <policy_config> <gpu_id>
# e.g. bash eval_policy.sh lift_bottle demo ACT_Modified/deploy_policy 0
```