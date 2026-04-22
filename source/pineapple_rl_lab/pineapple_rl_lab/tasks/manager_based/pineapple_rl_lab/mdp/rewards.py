# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import ContactSensor, RayCaster

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


def action_smoothness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float = 200.0,
) -> torch.Tensor:
    """Second-order action smoothness penalty: ||a_t - 2 a_{t-1} + a_{t-2}||^2."""
    # Ensure a two-step history is available
    if not hasattr(env, "_prev_prev_action") or env._prev_prev_action is None:
        # Initialize on first call
        env._prev_prev_action = torch.zeros_like(env.action_manager.prev_action)

    a_t = torch.nan_to_num(env.action_manager.action[:, asset_cfg.joint_ids], nan=0.0, posinf=0.0, neginf=0.0)
    a_t1 = torch.nan_to_num(env.action_manager.prev_action[:, asset_cfg.joint_ids], nan=0.0, posinf=0.0, neginf=0.0)
    a_t2 = torch.nan_to_num(env._prev_prev_action[:, asset_cfg.joint_ids], nan=0.0, posinf=0.0, neginf=0.0)

    diff2 = a_t - 2.0 * a_t1 + a_t2
    penalty = torch.sum(torch.square(diff2), dim=1)
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=max_penalty, neginf=0.0)
    penalty = torch.clamp(penalty, min=0.0, max=max_penalty)

    # Shift history for next step
    env._prev_prev_action = a_t1.clone()

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
    min_timestep: int = 0
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
             ang_acc_b = asset.data.body_ancc_b[:,0,:]
        else:
             # Transform world angular acceleration to body frame
             # alpha_b = R^T * alpha_w
             ang_acc_w = asset.data.body_ang_acc_w[:,0,:]
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

    if min_timestep > 0:
        active = (env.episode_length_buf >= min_timestep).float()
        reward = reward * active

    if return_debug_info:
        return reward, b3_C, z_body, quat_w, lin_acc_w, ang_vel_b, ang_acc_b 
    
    return reward


def track_base_height_exp(
    env: ManagerBasedRLEnv, 
    std: float,
    target_height: float | None = None,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking base height using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    if target_height is not None:
        target = torch.tensor(target_height, device=env.device)
    elif command_name is not None: 
         target = env.command_manager.get_command(command_name)[:, 0]
    else:
        raise ValueError("Either 'target_height' or 'command_name' must be provided.")
    
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        hit_z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        adjusted_target_height = target + torch.mean(hit_z, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target
    
    error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-error / std**2)


def front_obstacle_feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    vertical_height_threshold: float,
    command_threshold: float = 0.10,
) -> torch.Tensor:
    """Feet air-time reward activated when nearby terrain rise is detected.

    The reward is identical to feet-air-time-positive-biped, but it is gated by
    terrain rise from height-scanner ray hits.

    Args:
        vertical_height_threshold: Minimum rise to activate reward.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene[height_sensor_cfg.name]

    # --- Base feet air-time reward (biped single-stance style) ---
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    feet_air_reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    feet_air_reward = torch.clamp(feet_air_reward, max=threshold)

    # --- Front obstacle detection from height scanner ---
    ray_hits_w = height_sensor.data.ray_hits_w
    hit_z = ray_hits_w[..., 2]
    ground_height = torch.mean(hit_z, dim=1)
    obstacle_height = hit_z - ground_height.unsqueeze(-1)
    max_obstacle_height = torch.max(obstacle_height, dim=1).values
    terrain_mask = max_obstacle_height > vertical_height_threshold

    # No reward if command is too small.
    cmd_mask = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold

    active_mask = torch.logical_and(terrain_mask, cmd_mask).float()
    return feet_air_reward * active_mask


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    vertical_height_threshold: float | None = None,
    command_threshold: float = 0.10,
) -> torch.Tensor:
    """Reward swinging feet for clearing a target height above terrain.

    If ``sensor_cfg`` is provided, terrain height is estimated from the height scanner
    ray hits and the clearance target becomes ``terrain_height + target_height``.
    Otherwise, behavior falls back to the previous flat-ground formulation where
    ``target_height`` is interpreted in world z.

    Optional nearby-obstacle gating can be enabled by setting
    ``vertical_height_threshold``. In this mode, reward is only active when a
    nearby terrain rise is detected around the robot (not front-only).
    If ``command_name`` is provided, command gating is applied.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    terrain_mask = None
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_w = sensor.data.ray_hits_w
        terrain_height = torch.mean(ray_hits_w[..., 2], dim=1)
        # target_height_w = terrain_height.unsqueeze(1) + target_height
        target_height_w = target_height

        # Nearby obstacle gating, mirroring front_obstacle_feet_air_time_positive_biped.
        if vertical_height_threshold is not None:
            hit_z = ray_hits_w[..., 2]
            ground_height = torch.mean(hit_z, dim=1)
            obstacle_height = hit_z - ground_height.unsqueeze(-1)
            max_obstacle_height = torch.max(obstacle_height, dim=1).values
            terrain_mask = max_obstacle_height > vertical_height_threshold
    else:
        target_height_w = target_height

    foot_z_target_error = torch.square(foot_z - target_height_w)
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_xy, dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(reward, dim=1) / std)

    if terrain_mask is not None:
        if command_name is not None:
            cmd_mask = (
                torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
                > command_threshold
            )
            active_mask = torch.logical_and(terrain_mask, cmd_mask).float()
        else:
            active_mask = terrain_mask.float()
        reward = reward * active_mask

    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def lock_wheel_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    vertical_height_threshold: float = 0.01,
    command_threshold: float = 0.10,
) -> torch.Tensor:
    """Penalize wheel speed when obstacle-like terrain is detected."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(wheel_vel), dim=1)

    if sensor_cfg is None:
        return penalty

    sensor: RayCaster = env.scene[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    ground_height = torch.mean(hit_z, dim=1)
    obstacle_height = hit_z - ground_height.unsqueeze(-1)
    max_obstacle_height = torch.max(obstacle_height, dim=1).values
    terrain_mask = max_obstacle_height > vertical_height_threshold

    if command_name is not None:
        cmd_mask = (
            torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
            > command_threshold
        )
        terrain_mask = torch.logical_and(terrain_mask, cmd_mask)

    return penalty # * terrain_mask.float()

def no_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def gait_timing_biped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    std: float = 0.2,
    max_err: float = 0.5,
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Encourage left-right anti-phase gait using contact/air timers.

    For a biped, one foot in stance should correspond to the other in swing.
    This term avoids explicit phase signals and only uses contact sensor timers.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("gait_timing_biped expects exactly 2 bodies in sensor_cfg.body_ids.")

    left_id = sensor_cfg.body_ids[0]
    right_id = sensor_cfg.body_ids[1]

    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time

    # Anti-phase objective:
    # left-air ~= right-contact and left-contact ~= right-air.
    se_async_0 = torch.clip(torch.square(air_time[:, left_id] - contact_time[:, right_id]), max=max_err**2)
    se_async_1 = torch.clip(torch.square(contact_time[:, left_id] - air_time[:, right_id]), max=max_err**2)
    reward = torch.exp(-(se_async_0 + se_async_1) / std)

    # Enforce gait only while the robot is commanded (or already) to move.
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_speed = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    active = torch.logical_or(cmd_speed > command_threshold, body_speed > velocity_threshold)
    return torch.where(active, reward, 0.0)

def gait_phase_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, stance_threshold: float = 0.55) -> torch.Tensor:
    """Reward for foot contact matching the expected gait phase.
    
    Rewards the robot when foot contact status matches the expected stance/swing phase.
    During stance phase (phase < stance_threshold), foot should be in contact.
    During swing phase (phase >= stance_threshold), foot should be in the air.
    
    Args:
        env: Environment with gait phase information.
        sensor_cfg: Contact sensor configuration for feet.
        stance_threshold: Phase threshold below which the foot should be in stance.
        
    Reference: DreamWaQ _reward_contact()
    
    Note: This function uses env.leg_phase which should be [num_envs, num_feet] tensor
    where leg_phase[:, 0] = phase_left and leg_phase[:, 1] = phase_right.
    The sensor_cfg.body_ids should match the same ordering (left foot first, right foot second).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # Check contact for each foot (use z-component like original DreamWaQ)
    # Original: contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    contact = net_contact_forces[:, :, 2] > 1.0  # (num_envs, num_feet), z-direction force
    
    # Use leg_phase directly from environment
    # leg_phase shape: (num_envs, 2) where [:, 0] = left, [:, 1] = right
    leg_phase = env.leg_phase
    
    # Expected stance: phase < stance_threshold
    is_stance = leg_phase < stance_threshold
    
    # Reward: 1 if contact matches expected phase, 0 otherwise
    # XOR gives True when they don't match, so we negate it
    phase_match = ~(contact ^ is_stance)  # (num_envs, num_feet)
    
    return torch.sum(phase_match.float(), dim=-1)  # Sum over feet

def feet_swing_height(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08
) -> torch.Tensor:
    """Simple version: Penalize swing foot height deviation from fixed target.
    
    This is the original simple implementation that uses absolute z-coordinate.
    Use feet_swing_height() for terrain-aware version.
    
    Args:
        env: Environment.
        sensor_cfg: Contact sensor configuration for feet.
        asset_cfg: Robot configuration with body_ids for feet.
        target_height: Target height for swing foot (default 0.08m).
        
    Reference: DreamWaQ _reward_feet_swing_height()
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, num_feet)
    
    # Get feet positions (z-coordinate)
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
    
    # Penalize height error only during swing phase (not in contact)
    pos_error = torch.square(feet_pos_z - target_height) * (~contact).float()
    
    return torch.sum(pos_error, dim=-1)

def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)