# Pineapple RL IsaacLab

## Overview

This repository provides the Pineapple RL environment framework based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This project is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, pineapple, isaaclab

## Project Structure

This repository is organized as follows:

```text
source/pineapple_rl_lab/pineapple_rl_lab/
├── assets/
│   ├── assets/robots/
│   │   └── cslrobotics.py              # Robot Configurations (spawn settings, actuators)
│   └── data/Robots/csl/                # Robot Assets (URDFs, meshes)
│       ├── pineapplev0_description/    # v0 Model
│       └── pineapple/                  # v1 Model
└── tasks/manager_based/pineapple_rl_lab/
    ├── agents/
    │   └── rsl_rl_ppo_cfg.py           # RL Agent Configuration (PPO hyperparameters)
    ├── __init__.py # Environment registration
    ├── pineapple_rl_lab_env_cfg_v0.py  # Environment Configuration for v0
    └──pineapple_rl_lab_env_cfg_v1.py  # Environment Configuration for v1
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal. 
    
    Note: Code tested with `IsaacSim 5.1` and `IsaacLab 2.3.0`

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/pineapple_rl_lab
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        python scripts/rsl_rl/train.py --task=Template-Pineapple-Rl-Lab-v0 
        ```
- Train and replay policies:
    - Train a policy:

        ```bash
        python scripts/rsl_rl/train.py --task=Template-Pineapple-Rl-Lab-v0 --headless
        ```
    - Replay a policy:

        ```bash
        python scripts/rsl_rl/play.py --task=Template-Pineapple-Rl-Lab-Play-v0
        ```
    - Look for training logs:

        ```bash
        tensorboard --logdir <to/your/log/folder>
        ```



## Available Tasks

The following tasks are available for the Pineapple robot. Each task corresponds to a different version of the robot or a specific configuration (e.g., for training or play).

| Task Name | Description |
| :--- | :--- |
| `Template-Pineapple-Rl-Lab-v0` | Velocity tracking locomotion for Pineapple v0 on flat terrain. |
| `Template-Pineapple-Rl-Lab-v1` | Velocity tracking locomotion for Pineapple v1 on flat terrain. |
| `Template-Pineapple-Rl-Lab-Play-v0` | Play/Evaluation environment for Pineapple v0. |
| `Template-Pineapple-Rl-Lab-Play-v1` | Play/Evaluation environment for Pineapple v1. |


### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "/path/to/pineapple_rl_isaaclab/source/pineapple_rl_lab"
        "/path/to/IsaacLab/source/isaaclab_tasks",
        "/path/to//IsaacLab/source/isaaclab_rl",
        "/path/to//IsaacLab/source/isaaclab_assets",
        "/path/to//IsaacLab/source/isaaclab_mimic",
        "/path/to/IsaacLab/source/isaaclab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```