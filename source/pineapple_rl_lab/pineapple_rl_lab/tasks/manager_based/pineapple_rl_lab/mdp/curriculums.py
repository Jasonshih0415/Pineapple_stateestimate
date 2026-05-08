# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    reward_term_name: str,
    x_vel_range: tuple[float, float],
    z_ang_vel_range: tuple[float, float],
    reward_threshold_ratio: float = 0.8,
    range_step_size: float = 0.1,
) -> torch.Tensor:
    """Increases command range when the average tracking reward exceeds a threshold.

    This function checks the average episode reward for a specified tracking term
    for environments that are resetting. If the average reward is high enough,
    it increases the command range for ALL environments by a discrete step.

    Args:
        env: The environment instance.
        env_ids: The list of environment IDs that are resetting.
        command_name: The name of the velocity command to modify.
        reward_term_name: The name of the reward term to monitor (e.g., "tracking_lin_vel").
        x_vel_range: The final target range for the linear velocity in x.
        z_ang_vel_range: The final target range for the angular velocity in z.
        reward_threshold_ratio: The performance threshold (0-1) to trigger a difficulty increase.
        range_step_size: The fixed amount to increase the command range at each step.

    Returns:
        A tensor containing the current maximum linear velocity command, for logging.
    """
    # This curriculum only acts when some environments are resetting
    if len(env_ids) == 0:
        # Return the current max command range for logging
        command_term = env.command_manager.get_term(command_name)
        return torch.tensor(command_term.cfg.ranges.lin_vel_x[1], device=env.device)

    # Extract the used quantities
    command_term: UniformVelocityCommand = env.command_manager.get_term(command_name)
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    # Calculate the maximum possible reward for the episode for the tracked term
    # This assumes the reward is scaled and not shaped by a complex function
    max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s

    # Calculate the average reward for the episodes that just ended
    mean_episode_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])

    # low_vel_env_ids = (env_ids > (env.num_envs * 0.2))
    # high_vel_env_ids = (env_ids < (env.num_envs * 0.2))
    # low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
    # high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
    # low_mean_episode_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids])
    # high_mean_episode_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids])

    global_step = env.common_step_counter
    last_update_step = getattr(env, "_vel_curriculum_last_step", 0)

    # Check if the performance threshold is met
    # if (mean_episode_reward / max_possible_reward > reward_threshold_ratio): #and (high_mean_episode_reward / max_possible_reward > reward_threshold_ratio):
    if (
            global_step > 2000
            and global_step - last_update_step >= 100
            and mean_episode_reward / max_possible_reward > reward_threshold_ratio
        ):        
        setattr(env, "_vel_curriculum_last_step", global_step)
        current_ranges = command_term.cfg.ranges
        # Increase linear velocity range, clipping at the target maximum
        new_x_max = min(current_ranges.lin_vel_x[1] + range_step_size, x_vel_range[1])
        new_x_min = max(current_ranges.lin_vel_x[0] - range_step_size, x_vel_range[0])
        command_term.cfg.ranges.lin_vel_x = (new_x_min, new_x_max)

        # Increase angular velocity range, clipping at the target maximum
        new_z_max = min(current_ranges.ang_vel_z[1] + range_step_size, z_ang_vel_range[1])
        new_z_min = max(current_ranges.ang_vel_z[0] - range_step_size, z_ang_vel_range[0])
        command_term.cfg.ranges.ang_vel_z = (new_z_min, new_z_max)

    # Return the current max command range for logging purposes
    return torch.tensor(command_term.cfg.ranges.lin_vel_x[1], device=env.device)


def pace_motor_delay_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_names: list[str] | None = None,
    max_delay: int = 3,
    distance_thresholds: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> torch.Tensor:
    """Curriculum that increases PaceDCMotor action delay from 0 to max_delay based on episode distance.

    At each reset the average distance walked over the just-finished episodes is computed.
    When that distance crosses successive thresholds the global time-lag of every listed
    actuator is incremented by one step (up to ``max_delay``).

    .. note::
        The robot articulation must be initialised with ``max_delay`` equal to the final
        target value so that the ``DelayBuffer`` is allocated with sufficient capacity.
        Only the *active* time-lag is changed here, not the buffer size.

    Args:
        env: The environment instance.
        env_ids: Environment IDs that are resetting this step.
        asset_cfg: Scene entity config for the robot articulation.
        actuator_names: Names of the PaceDCMotor actuators to update.
                        If ``None``, all actuators on the asset are targeted.
        max_delay: Maximum time-lag (steps) to ramp up to.
        distance_thresholds: Distances (m) at which the delay is incremented by 1.
                             Must have exactly ``max_delay`` entries.

    Returns:
        Tensor containing the current active delay (for logging).
    """
    if len(env_ids) == 0:
        current_delay = getattr(env, "_pace_motor_current_delay", 0)
        return torch.tensor(current_delay, dtype=torch.float32, device=env.device)

    if len(distance_thresholds) != max_delay:
        raise ValueError(
            f"distance_thresholds must have exactly max_delay={max_delay} entries, "
            f"got {len(distance_thresholds)}."
        )

    asset: Articulation = env.scene[asset_cfg.name]

    # Distance walked this episode for each resetting env (same as terrain_levels_vel)
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )

    current_max_delay = getattr(env, "_pace_motor_current_delay", 0)

    if current_max_delay < max_delay:
        next_threshold = distance_thresholds[current_max_delay]
        # fraction of resetting envs that walked far enough to unlock the next level
        move_up = (distance > next_threshold).float().mean().item()
        if move_up > 0.8:
            current_max_delay += 1
            setattr(env, "_pace_motor_current_delay", current_max_delay)
            print(f"[pace_motor_delay_curriculum] max delay -> {current_max_delay} (move_up_frac={move_up:.2f})")

    # Sample a new per-env delay only for environments in this reset batch.
    # Other environments keep their current delay until their own reset, so their active lag
    # does not change mid-episode when an unrelated environment resets.
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    random_delays = torch.randint(0, current_max_delay + 1, (len(env_ids_tensor),), device=env.device).int()

    robot_actuators = asset.actuators
    targets = actuator_names if actuator_names is not None else list(robot_actuators.keys())
    for name in targets:
        if name in robot_actuators:
            robot_actuators[name].update_time_lags(random_delays, env_ids_tensor)
            # Read back to confirm
            actual = robot_actuators[name].torques_delay_buffer.time_lags[env_ids_tensor]
            if not (actual == random_delays).all():
                raise RuntimeError(
                    f"[pace_motor_delay_curriculum] Actuator '{name}': delay assignment mismatch."
                )

    return torch.tensor(float(current_max_delay), device=env.device)
