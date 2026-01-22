# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

## pineapple v0 robot
gym.register(
    id="Template-Pineapple-Rl-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v0:PineappleFlatEnvCfgV0",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Pineapple-Rl-Lab-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v0:PineappleFlatEnvCfgV0_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatPPORunnerCfg",
    },
)

## pineapple v1 robot
gym.register(
    id="Template-Pineapple-Rl-Lab-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v1:PineappleFlatEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Pineapple-Rl-Lab-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v1:PineappleFlatEnvCfgV1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatPPORunnerCfg",
    },
)

## pineapple v2 robot
# gym.register(
#     id="Template-Pineapple-Rl-Lab-v2",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2:PineappleFlatEnvCfgV2",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Template-Pineapple-Rl-Lab-Play-v2",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2:PineappleFlatEnvCfgV2_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatPPORunnerCfg",
#     },
# )