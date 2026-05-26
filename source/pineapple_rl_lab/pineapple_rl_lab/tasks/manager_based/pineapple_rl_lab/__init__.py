# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from . import agents_go2arm_style

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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV0PPORunnerCfg",
    },
)

gym.register(
    id="Template-Pineapple-Rl-Lab-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v0:PineappleFlatEnvCfgV0_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV0PPORunnerCfg",
    },
)

## pineapple v1 robot
gym.register(
    id="Template-Pineapple-Rl-Lab-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v1:PineappleFlatEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV1PPORunnerCfg",
    },
)

gym.register(
    id="Template-Pineapple-Rl-Lab-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v1:PineappleFlatEnvCfgV1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV1PPORunnerCfg",
    },
)

## pineapple v2 robot - x linear velocity command + z angular velocity command, no height command
gym.register(
    id="Template-Pineapple-Rl-Lab-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2:PineappleFlatEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV2PPORunnerCfg",
    },
)

gym.register(
    id="Template-Pineapple-Rl-Lab-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2:PineappleFlatEnvCfgV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV2PPORunnerCfg",
    },
)

## pineapple v2 robot - x linear velocity command + z angular velocity command + height command
gym.register(
    id="Template-Pineapple-Rl-Lab-v2.1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_1:PineappleFlatEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV2PPORunnerCfg",
    },
)

gym.register(
    id="Template-Pineapple-Rl-Lab-Play-v2.1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_1:PineappleFlatEnvCfgV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleFlatV2PPORunnerCfg",
    },
)

# ## pineapple v2 robot - rough terrain with curriculum
# gym.register(
#     id="Template-Pineapple-Rl-Lab-v2.2",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_2:PineappleRoughEnvCfgV2",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleRoughV2PPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="Template-Pineapple-Rl-Lab-Play-v2.2",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_2:PineappleRoughEnvCfgV2_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleRoughV2PPORunnerCfg",
#     },
# )
#
# ## pineapple v2 robot - rough terrain + DWAQ runner
# gym.register(
#     id="Template-Pineapple-Rl-Lab-DWAQ-v2.2",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_2_dwaq:PineappleRoughEnvCfgV2DWAQ",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleRoughV2DWAQRunnerCfg",
#     },
# )
#
# gym.register(
#     id="Template-Pineapple-Rl-Lab-DWAQ-Play-v2.2",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_2_dwaq:PineappleRoughEnvCfgV2DWAQ_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleRoughV2DWAQRunnerCfg",
#     },
# )
#
# ## pineapple v2 robot - rough terrain + DWAQ runner + fix wheels
# gym.register(
#     id="Template-Pineapple-Rl-Lab-DWAQ-v2.3",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_3_dwaq:PineappleRoughEnvCfgV2DWAQ",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleRoughV2DWAQRunnerCfg",
#     },
# )
#
# gym.register(
#     id="Template-Pineapple-Rl-Lab-DWAQ-Play-v2.3",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pineapple_rl_lab_env_cfg_v2_3_dwaq:PineappleRoughEnvCfgV2DWAQ_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PineappleRoughV2DWAQRunnerCfg",
#     },
# )

gym.register(
    id="Template-Pineapple-Arm-Rl-Lab-v2.5",
    entry_point="pineapple_rl_lab.env_go2arm_style.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__package__}.pineapple_rl_lab_env_cfg_v2_5:PineappleFlatEnvCfgV25",
        "rsl_rl_cfg_entry_point": (
            f"{agents_go2arm_style.__name__}.rsl_rl_ppo_cfg_v2_5:PineappleArmV25PPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Pineapple-Arm-Rl-Lab-Play-v2.5",
    entry_point="pineapple_rl_lab.env_go2arm_style.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__package__}.pineapple_rl_lab_env_cfg_v2_5:PineappleFlatEnvCfgV25_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents_go2arm_style.__name__}.rsl_rl_ppo_cfg_v2_5:PineappleArmV25PPORunnerCfg"
        ),
    },
)
