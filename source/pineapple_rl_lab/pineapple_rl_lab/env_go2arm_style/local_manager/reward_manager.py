from __future__ import annotations
import torch
from isaaclab.managers import RewardManager as RewardManagerBase

class RewardManager(RewardManagerBase):

    def __init__(self,cfg, env):
        super().__init__(cfg, env)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 
        self._log_arm_reward_joint_resolution()

    def _log_arm_reward_joint_resolution(self):
        """Print the resolved joints for arm-specific joint reward terms once at startup."""
        target_terms = {
            "end_effector_joint_vel",
            "end_effector_joint_torques",
            "end_effector_joint_acc",
        }
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if name not in target_terms or "asset_cfg" not in term_cfg.params:
                continue

            asset_cfg = term_cfg.params["asset_cfg"]
            asset = self._env.scene[asset_cfg.name]
            joint_ids = asset_cfg.joint_ids
            joint_names = getattr(asset, "joint_names", None)
            if joint_names is None:
                joint_names = getattr(asset.data, "joint_names", None)

            if isinstance(joint_ids, slice):
                resolved_ids = list(range(len(joint_names))) if joint_names is not None else joint_ids
            elif isinstance(joint_ids, torch.Tensor):
                resolved_ids = joint_ids.detach().cpu().tolist()
            else:
                resolved_ids = list(joint_ids)

            if joint_names is not None and not isinstance(resolved_ids, slice):
                resolved_names = [joint_names[joint_id] for joint_id in resolved_ids]
            else:
                resolved_names = "<unavailable>"

            print(f"[INFO] Reward term '{name}' resolved joint_ids={resolved_ids}, joint_names={resolved_names}")


    def compute(self, dt: float) -> tuple[torch.Tensor,torch.Tensor]:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        self.arm_reward_buf[:] = 0.0 
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # check if the term is a special term for arm
            if name.startswith("end_effector"):  ## TODO: 
                self.arm_reward_buf += value
            else:
                self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf, self.arm_reward_buf
