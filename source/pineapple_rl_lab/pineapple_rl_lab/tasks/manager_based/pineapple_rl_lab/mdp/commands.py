"""Sub-module containing command generators for the environment."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PineappleVelocityCommand(UniformVelocityCommand):
    """Command generator that samples velocity commands with coupled limits.
    
    This command generator enforces a diamond-shaped constraint (L1 norm) on the linear and angular velocities:
    |v_x| / v_max + |w_z| / w_max <= 1
    
    This allows larger individual velocities when the other component is small, 
    while restricting the combination to avoid exceeding wheel speed limits.
    """

    cfg: PineappleVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment IDs."""
        # Convert env_ids to tensor if needed for consistent indexing
        if not isinstance(env_ids, torch.Tensor):
            ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            ids = env_ids

        len_ids = len(ids)
        r = torch.rand(len_ids, device=self.device)
        
        # Sample Linear Velocity X
        # The cfg ranges are tuples (min, max).
        min_vx, max_vx_val = self.cfg.ranges.lin_vel_x
        self.vel_command_b[ids, 0] = r * (max_vx_val - min_vx) + min_vx
        
        # Sample Linear Velocity Y
        r = torch.rand(len_ids, device=self.device)
        min_vy, max_vy_val = self.cfg.ranges.lin_vel_y
        self.vel_command_b[ids, 1] = r * (max_vy_val - min_vy) + min_vy
        
        # Sample Angular Velocity Z
        r = torch.rand(len_ids, device=self.device)
        min_wz, max_wz_val = self.cfg.ranges.ang_vel_z
        self.vel_command_b[ids, 2] = r * (max_wz_val - min_wz) + min_wz
        
        # --- Apply Coupled L1 Norm Constraint ---
        # Calculate max magnitudes for normalization (intercepts)
        # We use the configured ranges as the maximum possible values (intercepts of the diamond).
        limit_vx = max(abs(min_vx), abs(max_vx_val))
        limit_wz = max(abs(min_wz), abs(max_wz_val))
        
        # Avoid division by zero
        if limit_vx < 1e-6: limit_vx = 1.0
        if limit_wz < 1e-6: limit_wz = 1.0
        
        # Compute the constraint ratio: |v_x|/limit_vx + |w_z|/limit_wz
        vx = self.vel_command_b[ids, 0]
        wz = self.vel_command_b[ids, 2]
        
        ratio = torch.abs(vx) / limit_vx + torch.abs(wz) / limit_wz
        
        # Identify violations
        violation_mask = ratio > 1.0
        
        # Project violations onto the boundary of the diamond
        # We scale both components by 1/ratio, which preserves the direction (ratio of vx to wz)
        if torch.any(violation_mask):
            scaling = 1.0 / ratio[violation_mask]
            
            # Get the indices of environments with violations
            bad_indices = ids[violation_mask]
            
            # Apply scaling
            self.vel_command_b[bad_indices, 0] *= scaling
            self.vel_command_b[bad_indices, 2] *= scaling
            
        # --- Heading Logic ---
        if self.cfg.heading_command:
            r = torch.rand(len_ids, device=self.device)
            min_h, max_h = self.cfg.ranges.heading
            self.heading_target[ids] = r * (max_h - min_h) + min_h
            
            # Update heading envs probability
            r = torch.rand(len_ids, device=self.device)
            self.is_heading_env[ids] = r <= self.cfg.rel_heading_envs

        # --- Standing Env Logic ---
        r = torch.rand(len_ids, device=self.device)
        self.is_standing_env[ids] = r <= self.cfg.rel_standing_envs


@configclass
class PineappleVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the pineapple velocity command generator."""

    class_type: type = PineappleVelocityCommand


class PineappleHeightCommand(CommandTerm):
    """Command generator that samples height commands."""

    cfg: PineappleHeightCommandCfg

    def __init__(self, cfg: PineappleHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics = {}

    def _resample_command(self, env_ids: Sequence[int]):
        # sample height
        # Convert env_ids to tensor if needed for consistent indexing
        if not isinstance(env_ids, torch.Tensor):
            ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            ids = env_ids
        
        r = torch.rand(len(ids), device=self.device)
        min_h, max_h = self.cfg.ranges.height
        self.height_command[ids, 0] = r * (max_h - min_h) + min_h

        # Apply default height probability
        if self.cfg.rel_default_height_envs > 0.0:
            r_default = torch.rand(len(ids), device=self.device)
            is_default = r_default <= self.cfg.rel_default_height_envs
            # Use default_height if set, otherwise use middle of range (or 0.3 as fallback)
            default_h = self.cfg.default_height if self.cfg.default_height is not None else 0.3
            self.height_command[ids[is_default], 0] = default_h

    def _update_command(self):
        pass
    
    def _update_metrics(self):
        pass

    @property
    def command(self):
        return self.height_command


@configclass
class PineappleHeightCommandCfg(CommandTermCfg):
    """Configuration for the pineapple height command generator."""
    class_type: type = PineappleHeightCommand
    
    asset_name: str = MISSING

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    default_height: float = 0.3
    rel_default_height_envs: float = 0.0
