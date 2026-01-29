# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# def joint_position_penalty(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
# ) -> torch.Tensor:
#     """Penalize joint position error from default on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#     reward = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
#     return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


# def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint velocities on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.linalg.norm((asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_smoothness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Second-order action smoothness penalty: ||a_t - 2 a_{t-1} + a_{t-2}||^2."""
    # Ensure a two-step history is available
    if not hasattr(env, "_prev_prev_action") or env._prev_prev_action is None:
        # Initialize on first call
        env._prev_prev_action = torch.zeros_like(env.action_manager.prev_action)

    a_t  = env.action_manager.action[:, asset_cfg.joint_ids]                # current action
    a_t1 = env.action_manager.prev_action[:, asset_cfg.joint_ids]           # last action
    a_t2 = env._prev_prev_action[:, asset_cfg.joint_ids]                    # last-last action

    diff2 = a_t - 2.0 * a_t1 + a_t2
    penalty = torch.sum(torch.square(diff2), dim=1)

    # Shift history for next step
    env._prev_prev_action = env.action_manager.prev_action.clone()

    return penalty


def joint_align(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_a: str,
    joint_b: str,
    use: str = "pos",  # "pos", "vel", or "action"
) -> torch.Tensor:
    """Return |joint_a - joint_b| per env."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve indices (consistent with existing code style)
    idx_a = asset.find_joints(joint_a)[0]
    idx_b = asset.find_joints(joint_b)[0]

    if use == "pos":
        a = asset.data.joint_pos[:, idx_a]
        b = asset.data.joint_pos[:, idx_b]
    elif use == "vel":
        a = asset.data.joint_vel[:, idx_a]
        b = asset.data.joint_vel[:, idx_b]
    elif use == "action":
        a = env.action_manager.action[:, idx_a]
        b = env.action_manager.action[:, idx_b]
    else:
        raise ValueError("use must be one of: 'pos', 'vel', 'action'.")

    return torch.sum(torch.abs(a-b), dim=1)

# def base_height_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
#     """Penalize deviation from target base height."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     return torch.square(asset.data.root_pos_w[:, 2] - target_height)
