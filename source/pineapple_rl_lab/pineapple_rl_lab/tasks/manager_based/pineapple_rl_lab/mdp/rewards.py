# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

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


def slosh_free(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    r_offset: list[float] | None = None,
    g: float = 9.81,
    use_finite_diff: bool = False,
    internal_state_suffix: str = "",
    return_debug_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reward based on 'slosh-free' condition: aligning apparent gravity with container's vertical axis.
    
    Args:
        env: The environment.
        asset_cfg: The configuration for the asset (e.g., robot base).
        r_offset: Vector from Body COM to the fluid surface/sensor in body frame.
                  Defaults to [0.0, 0.0, 0.0].
        g: Gravity magnitude.
        use_finite_diff: If True, force finite difference calculation for acceleration.
                         If False, attempt to use direct acceleration from asset data if available.
        internal_state_suffix: Suffix for internal state attributes to avoid collision during debug.
        return_debug_info: If True, returns (reward, b3_C, z_body, aC).
    """
    if r_offset is None:
        r_offset = [0.0, 0.0, 0.0]
    
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Current state
    lin_vel_w = asset.data.root_lin_vel_w  # World frame linear velocity
    ang_vel_b = asset.data.root_ang_vel_b  # Body frame angular velocity
    quat_w = asset.data.root_quat_w        # World frame orientation (w, x, y, z usually in Isaac Lab)
    
    # Internal state names
    prev_lin_name = f"_slosh_prev_lin_vel_w{internal_state_suffix}"
    prev_ang_name = f"_slosh_prev_ang_vel_b{internal_state_suffix}"

    # Initialize history buffers if needed (always do this to be safe or if switching methods)
    if not hasattr(env, prev_lin_name):
        setattr(env, prev_lin_name, torch.zeros_like(lin_vel_w))
        setattr(env, prev_ang_name, torch.zeros_like(ang_vel_b))

    dt = env.step_dt

    # Try to get direct acceleration if requested and available
    has_direct_lin_acc = hasattr(asset.data, "body_lin_acc_w")
    # Check for body or world angular acceleration
    has_direct_ang_acc_b = hasattr(asset.data, "body_ang_acc_b")
    has_direct_ang_acc_w = hasattr(asset.data, "body_ang_acc_w")

    # Manual implementation of quat to rot matrix for batch
    # q = [w, x, y, z]
    w, x, y, z = quat_w.unbind(dim=-1)
    
    # First row of the rotation matrix
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    
    # Second row of the rotation matrix
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    
    # Third row of the rotation matrix
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    
    # Stack to form R (Batch x 3 x 3)
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    if not use_finite_diff and has_direct_lin_acc and (has_direct_ang_acc_b or has_direct_ang_acc_w):
        # Use direct acceleration
        lin_acc_w = asset.data.body_lin_acc_w[:,0,:]
        
        if has_direct_ang_acc_b:
             ang_acc_b = asset.data.body_ancc_b[:,0:]
        else:
             # Transform world angular acceleration to body frame
             # alpha_b = R^T * alpha_w
             ang_acc_w = asset.data.body_ang_acc_w[:,0:]
             # R is (Batch, 3, 3). ang_acc_w is (Batch, 3).
             # R^T is transpose of last two dims.
             # batch matmul: R^T @ ang_acc_w
             ang_acc_b = torch.bmm(R.transpose(-1, -2), ang_acc_w.unsqueeze(-1)).squeeze(-1)
             
        # Need to update history buffers anyway to keep them consistent if we switch back
        # But technically not needed for calculation.
        getattr(env, prev_lin_name)[:] = lin_vel_w
        getattr(env, prev_ang_name)[:] = ang_vel_b
    else:
        # Finite difference fallback
        prev_lin = getattr(env, prev_lin_name)
        prev_ang = getattr(env, prev_ang_name)
        
        lin_acc_w = (lin_vel_w - prev_lin) / dt
        ang_acc_b = (ang_vel_b - prev_ang) / dt

        # Update history buffers
        prev_lin[:] = lin_vel_w
        prev_ang[:] = ang_vel_b

    
    # Prepare r vector (Batch x 3)
    r_vec = torch.tensor(r_offset, device=lin_vel_w.device, dtype=lin_vel_w.dtype).expand(lin_vel_w.shape[0], 3)
    
    # Skew symmetric matrix of omega (Batch x 3 x 3)
    # [ 0 -z  y]
    # [ z  0 -x]
    # [-y  x  0]
    wx, wy, wz = ang_vel_b.unbind(dim=-1)
    zeros = torch.zeros_like(wx)
    skew_omega = torch.stack([
        torch.stack([zeros, -wz, wy], dim=-1),
        torch.stack([wz, zeros, -wx], dim=-1),
        torch.stack([-wy, wx, zeros], dim=-1)
    ], dim=-2)
    
    # Skew symmetric matrix of alpha (Batch x 3 x 3)
    ax, ay, az = ang_acc_b.unbind(dim=-1)
    skew_alpha = torch.stack([
        torch.stack([zeros, -az, ay], dim=-1),
        torch.stack([az, zeros, -ax], dim=-1),
        torch.stack([-ay, ax, zeros], dim=-1)
    ], dim=-2)
    
    # Terms for acceleration at point C
    # centripetal = omega x (omega x r) = skew(omega) @ (skew(omega) @ r)
    # tangential = alpha x r = skew(alpha) @ r
    # Note: Using matmul for batch matrix-vector multiplication (Batch x 3 x 3) @ (Batch x 3 x 1)
    
    r_vec_unsqueezed = r_vec.unsqueeze(-1) # (Batch x 3 x 1)
    
    term1 = torch.bmm(skew_omega, r_vec_unsqueezed) # (omega x r)
    centripetal = torch.bmm(skew_omega, term1).squeeze(-1) # omega x (omega x r)
    tangential = torch.bmm(skew_alpha, r_vec_unsqueezed).squeeze(-1) # alpha x r
    
    # a_container = a_W + R * (tangential + centripetal)
    # tangential + centripetal is in body frame. R transforms to world.
    body_acc_term = tangential + centripetal
    world_acc_term = torch.bmm(R, body_acc_term.unsqueeze(-1)).squeeze(-1)
    
    aC = lin_acc_w + world_acc_term
    
    # Apparent gravity vector (Thrust-like direction)
    # fb3_C = aC + [0.0, 0.0, g]
    gravity_vec = torch.tensor([0.0, 0.0, g], device=lin_vel_w.device, dtype=lin_vel_w.dtype).expand_as(aC)
    fb3_C = aC + gravity_vec
    
    mag = torch.linalg.norm(fb3_C, dim=1, keepdim=True)
    
    # Avoid division by zero
    mag = torch.where(mag < 1e-8, torch.ones_like(mag), mag)
    
    b3_C = fb3_C / mag      # Normalized apparent gravity direction in World Frame
    
    # z_body in world frame is the 3rd column of R
    z_body = R[:, :, 2] # (Batch x 3)
    
    # Error is 1 - cos(theta) = 1 - dot(b3_C, z_body)
    dot_prod = torch.sum(b3_C * z_body, dim=1)
    
    reward = 1.0 - dot_prod

    if return_debug_info:
        return reward, b3_C, z_body, aC
    
    return reward
