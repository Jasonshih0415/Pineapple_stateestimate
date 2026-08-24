from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from headless_teleop import HeadlessTeleop
from capo_estimator_bipedal import PineappleV2StateEstimator


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "pineapple_v2.yaml"
LOG_DIR = Path(__file__).resolve().parent / "logs"


@dataclass
class SimConfig:
    policy_path: str
    xml_path: str
    simulation_duration: float
    simulation_dt: float
    control_decimation: int
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    pos_action_scale: float
    vel_action_scale: float
    cmd_scale: np.ndarray
    num_actions: int
    num_obs: int
    one_step_obs_size: int
    obs_buffer_size: int
    leg_joint_indices: list[int]
    wheel_joint_indices: list[int]
    cmd_init: np.ndarray
    max_lin: float
    max_ang: float
    height_scale: float
    cmd_height_init: float
    min_height: float
    max_height: float
    height_step: float
    enable_height_command: bool
    policy_index_map: np.ndarray | None

    @classmethod
    def from_dict(cls, config: dict) -> "SimConfig":
        policy_index_map = config.get("policy_index_map", None)
        if policy_index_map is not None:
            policy_index_map = np.array(policy_index_map, dtype=np.int64)

        return cls(
            policy_path=config["policy_path"],
            xml_path=config["xml_path"],
            simulation_duration=config["simulation_duration"],
            simulation_dt=config["simulation_dt"],
            control_decimation=config["control_decimation"],
            kps=np.array(config["kps"], dtype=np.float32),
            kds=np.array(config["kds"], dtype=np.float32),
            default_angles=np.array(config["default_angles"], dtype=np.float32),
            lin_vel_scale=config["lin_vel_scale"],
            ang_vel_scale=config["ang_vel_scale"],
            dof_pos_scale=config["dof_pos_scale"],
            dof_vel_scale=config["dof_vel_scale"],
            pos_action_scale=config["pos_action_scale"],
            vel_action_scale=config["vel_action_scale"],
            cmd_scale=np.array(config["cmd_scale"], dtype=np.float32),
            num_actions=config["num_actions"],
            num_obs=config["num_obs"],
            one_step_obs_size=config["one_step_obs_size"],
            obs_buffer_size=config.get("obs_buffer_size", 1),
            leg_joint_indices=config["leg_joint_indices"],
            wheel_joint_indices=config["wheel_joint_indices"],
            cmd_init=np.array(config["cmd_init"], dtype=np.float32),
            max_lin=config.get("max_lin_vel", 1.0),
            max_ang=config.get("max_ang_vel", 1.0),
            height_scale=config.get("height_scale", 1.0),
            cmd_height_init=config.get("cmd_height_init", 0.3),
            min_height=config.get("min_height", 0.2),
            max_height=config.get("max_height", 0.35),
            height_step=config.get("height_step", 0.005),
            enable_height_command=config.get("enable_height_command", True),
            policy_index_map=policy_index_map,
        )


@dataclass
class HistoryBuffers:
    lin_vel: list[np.ndarray] = field(default_factory=list)
    ang_vel: list[np.ndarray] = field(default_factory=list)
    gravity_b: list[np.ndarray] = field(default_factory=list)
    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    time: list[float] = field(default_factory=list)
    cmd: list[np.ndarray] = field(default_factory=list)
    tau: list[np.ndarray] = field(default_factory=list)


@dataclass
class SipoBuffers:
    pos: list[np.ndarray] = field(default_factory=list)
    vel: list[np.ndarray] = field(default_factory=list)
    quat: list[np.ndarray] = field(default_factory=list)
    vel_body: list[np.ndarray] = field(default_factory=list)
    gt_pos: list[np.ndarray] = field(default_factory=list)
    gt_vel: list[np.ndarray] = field(default_factory=list)
    gt_quat: list[np.ndarray] = field(default_factory=list)
    gt_vel_body: list[np.ndarray] = field(default_factory=list)
    imu_yaw_rate: list[float] = field(default_factory=list)
    yaw_rate: list[float] = field(default_factory=list)
    gt_yaw_rate: list[float] = field(default_factory=list)
    wheel_radius: float = 0.0


@dataclass
class SipoRunner:
    sipo: Any
    get_contact_states: Callable
    buffers: SipoBuffers
    base_body_id: int

    @classmethod
    def create(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: SimConfig,
        base_body_id: int,
    ) -> "SipoRunner":
        # from mujoco_sipo_v3 import SIPO, get_contact_states
        from mujoco_sipo_v3_new import SIPO, get_contact_states
        sipo = SIPO(config.xml_path)
        mujoco.mj_forward(model, data)

        init_qpos_sense = data.qpos[7 : 7 + config.num_actions].copy()
        init_qvel_sense = data.qvel[6 : 6 + config.num_actions].copy()
        z_kin = sipo.get_kinematics(init_qpos_sense, init_qvel_sense)
        feet_pos_flat = z_kin.reshape(sipo.num_legs, sipo.fk_stride)[:, :3].flatten()

        sipo.init_state(
            data.xipos[base_body_id].copy(),
            data.xquat[base_body_id].copy(),
            feet_pos_flat,
        )
        print("SIPO initialized after reset.")

        buffers = SipoBuffers(wheel_radius=sipo.wheel_radius)
        return cls(sipo, get_contact_states, buffers, base_body_id)

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos,
        qvel,
        imu_acc,
        ang_vel_b,
        config: SimConfig,
    ) -> None:
        contacts = self.get_contact_states(model, data, self.sipo.leg_names)
        contacts.fill(1.0)

        self.sipo.predict(imu_acc, ang_vel_b, config.simulation_dt, qpos)
        wheel_vel_meas = qvel[config.wheel_joint_indices]
        sipo_state = self.sipo.update(
            qpos,
            qvel,
            contacts,
            ang_vel_b,
            wheel_vel_meas,
            yaw_meas=None,
        )

        buffers = self.buffers
        buffers.pos.append(sipo_state[self.sipo.idx_pos].copy())
        buffers.vel.append(sipo_state[self.sipo.idx_vel].copy())
        buffers.quat.append(sipo_state[self.sipo.idx_quat].copy())
        buffers.vel_body.append(
            quat_rotate_inverse(sipo_state[self.sipo.idx_quat], sipo_state[self.sipo.idx_vel])
        )
        buffers.gt_pos.append(data.xpos[self.base_body_id].copy())
        buffers.gt_vel.append(data.cvel[self.base_body_id][3:6].copy())
        buffers.gt_quat.append(data.xquat[self.base_body_id].copy())
        buffers.gt_vel_body.append(
            quat_rotate_inverse(data.xquat[self.base_body_id], data.cvel[self.base_body_id][3:6])
        )
        buffers.imu_yaw_rate.append(ang_vel_b[2])

        bg = sipo_state[self.sipo.idx_bg]
        buffers.yaw_rate.append(ang_vel_b[2] - bg[2])

        gt_w_world = data.cvel[self.base_body_id][0:3].copy()
        gt_w_body = quat_rotate_inverse(data.xquat[self.base_body_id], gt_w_world)
        buffers.gt_yaw_rate.append(gt_w_body[2])


@dataclass
class CapoCandidateBuffers:
    """Diagnostics for one wheel-sign/contact-threshold candidate."""

    wheel_sign: int
    threshold: float
    vel_body: list[np.ndarray] = field(default_factory=list)
    vel_body_raw: list[np.ndarray] = field(default_factory=list)
    gt_vel_body: list[np.ndarray] = field(default_factory=list)
    force_z: list[np.ndarray] = field(default_factory=list)
    contact_probability: list[np.ndarray] = field(default_factory=list)
    contact_est: list[np.ndarray] = field(default_factory=list)
    contact_gt: list[np.ndarray] = field(default_factory=list)
    wheel_dq: list[np.ndarray] = field(default_factory=list)


@dataclass
class CapoCandidate:
    estimator: PineappleV2StateEstimator
    buffers: CapoCandidateBuffers


class CapoDiagnosticRunner:
    """Run a grid of wheel-sign and contact-threshold CAPO candidates."""

    def __init__(
        self,
        model: mujoco.MjModel,
        thresholds: list[float],
        update_rate_hz: float,
    ) -> None:
        self.wheel_body_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_wheel"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_wheel"),
            ],
            dtype=np.int32,
        )
        if np.any(self.wheel_body_ids < 0):
            raise ValueError("CAPO diagnostics require L_wheel and R_wheel bodies")

        self.candidates: list[CapoCandidate] = []
        for wheel_sign in (1, -1):
            for threshold in thresholds:
                estimator = PineappleV2StateEstimator(
                    foot_force_threshold=threshold,
                    enable_leg_yaw=False,
                    enable_slope=False,
                    # Compare raw model behavior without candidate-specific
                    # scale or output-filter lag.
                    vel_filter_tau=0.0,
                    vel_median_window=1,
                    vel_scale=(1.0, 1.0, 1.0),
                    update_rate_hz=update_rate_hz,
                )
                buffers = CapoCandidateBuffers(wheel_sign, threshold)
                self.candidates.append(CapoCandidate(estimator, buffers))

    def _ground_truth_contacts(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> np.ndarray:
        contacts = np.zeros(2, dtype=bool)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            for wheel_index, wheel_body_id in enumerate(self.wheel_body_ids):
                if body1 == wheel_body_id or body2 == wheel_body_id:
                    contacts[wheel_index] = True
        return contacts

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos: np.ndarray,
        qvel: np.ndarray,
        tau_meas: np.ndarray,
        imu_quat: np.ndarray,
        ang_vel_b: np.ndarray,
        imu_acc: np.ndarray,
        gt_vel_world: np.ndarray,
        timestamp_ms: int,
    ) -> None:
        contact_gt = self._ground_truth_contacts(model, data)
        gt_vel_body = quat_rotate_inverse(imu_quat, gt_vel_world)

        for candidate in self.candidates:
            q_test = np.asarray(qpos, dtype=float).copy()
            dq_test = np.asarray(qvel, dtype=float).copy()
            # Wheel q/dq are used only by CAPO's rolling constraint, so this
            # tests encoder sign without changing leg FK or MuJoCo control.
            q_test[[3, 7]] *= candidate.buffers.wheel_sign
            dq_test[[3, 7]] *= candidate.buffers.wheel_sign

            result = candidate.estimator.update(
                imu_quat,
                ang_vel_b,
                imu_acc,
                q_test,
                dq_test,
                tau_meas,
                timestamp_ms,
            )
            legs = candidate.estimator.core.legs_pos
            force_z = np.array(
                [legs.FootBodyEff_WF[i][2] for i in range(2)], dtype=float
            )

            b = candidate.buffers
            b.vel_body.append(np.asarray(result.lin_vel_body, dtype=float))
            b.vel_body_raw.append(np.asarray(result.lin_vel_body_raw, dtype=float))
            b.gt_vel_body.append(np.asarray(gt_vel_body, dtype=float).copy())
            b.force_z.append(force_z)
            b.contact_probability.append(np.asarray(result.foot_contact, dtype=float))
            b.contact_est.append(force_z < b.threshold)
            b.contact_gt.append(contact_gt.copy())
            b.wheel_dq.append(np.asarray(qvel[[3, 7]], dtype=float).copy())


@dataclass
class CapoBuffers:
    pos: list[np.ndarray] = field(default_factory=list)
    vel_world: list[np.ndarray] = field(default_factory=list)
    vel_body: list[np.ndarray] = field(default_factory=list)
    rpy: list[np.ndarray] = field(default_factory=list)
    angular_velocity: list[np.ndarray] = field(default_factory=list)
    contacts: list[np.ndarray] = field(default_factory=list)
    force_z: list[np.ndarray] = field(default_factory=list)
    gt_pos: list[np.ndarray] = field(default_factory=list)
    gt_vel_world: list[np.ndarray] = field(default_factory=list)
    gt_vel_body: list[np.ndarray] = field(default_factory=list)
    gt_yaw_rate: list[float] = field(default_factory=list)
    imu_yaw_rate: list[float] = field(default_factory=list)


@dataclass
class CapoRunner:
    """Fixed Pineapple CAPO configuration used for normal estimation plots."""

    estimator: PineappleV2StateEstimator
    buffers: CapoBuffers
    base_body_id: int

    @classmethod
    def create(cls, config: SimConfig, base_body_id: int) -> "CapoRunner":
        estimator = PineappleV2StateEstimator(
            foot_force_threshold=-15.0,
            enable_leg_yaw=False,
            enable_slope=False,
            update_rate_hz=1.0 / config.simulation_dt,
        )
        return cls(estimator, CapoBuffers(), base_body_id)

    def update(
        self,
        data: mujoco.MjData,
        qpos: np.ndarray,
        qvel: np.ndarray,
        tau_meas: np.ndarray,
        imu_quat: np.ndarray,
        ang_vel_b: np.ndarray,
        imu_acc: np.ndarray,
        gt_vel_world: np.ndarray,
        timestamp_ms: int,
    ) -> None:
        result = self.estimator.update(
            imu_quat,
            ang_vel_b,
            imu_acc,
            qpos,
            qvel,
            tau_meas,
            timestamp_ms,
        )
        legs = self.estimator.core.legs_pos
        gt_vel_body = quat_rotate_inverse(imu_quat, gt_vel_world)
        gt_w_world = data.cvel[self.base_body_id][0:3].copy()
        gt_w_body = quat_rotate_inverse(data.xquat[self.base_body_id], gt_w_world)

        b = self.buffers
        b.pos.append(np.asarray(result.pos_world, dtype=float))
        b.vel_world.append(np.asarray(result.lin_vel_world, dtype=float))
        b.vel_body.append(np.asarray(result.lin_vel_body, dtype=float))
        b.rpy.append(np.asarray(result.rpy, dtype=float))
        b.angular_velocity.append(
            np.array(
                [result.odom.RollVel, result.odom.PitchVel, result.odom.YawVel],
                dtype=float,
            )
        )
        b.contacts.append(np.asarray(result.foot_contact, dtype=float))
        b.force_z.append(
            np.array([legs.FootBodyEff_WF[i][2] for i in range(2)], dtype=float)
        )
        b.gt_pos.append(data.xpos[self.base_body_id].copy())
        b.gt_vel_world.append(np.asarray(gt_vel_world, dtype=float).copy())
        b.gt_vel_body.append(np.asarray(gt_vel_body, dtype=float))
        b.gt_yaw_rate.append(float(gt_w_body[2]))
        b.imu_yaw_rate.append(float(ang_vel_b[2]))


@dataclass
class CommandStep:
    duration: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height: float | None = None


class CommandSequencer:
    """Replays a fixed timed sequence of velocity/height commands."""

    def __init__(self, steps: list[CommandStep], default_height: float):
        self.steps = steps
        self.default_height = default_height
        self._boundaries: list[float] = []
        t = 0.0
        for s in steps:
            t += s.duration
            self._boundaries.append(t)

    @property
    def total_duration(self) -> float:
        return self._boundaries[-1] if self._boundaries else 0.0

    def get_command(self, sim_time: float) -> tuple[np.ndarray, float]:
        for i, step in enumerate(self.steps):
            if sim_time < self._boundaries[i]:
                height = step.height if step.height is not None else self.default_height
                return np.array([step.vx, step.vy, step.wz], dtype=np.float32), height
        last = self.steps[-1]
        height = last.height if last.height is not None else self.default_height
        return np.array([last.vx, last.vy, last.wz], dtype=np.float32), height


def load_config(config_file: str) -> SimConfig:
    with open(config_file, "r") as f:
        return SimConfig.from_dict(yaml.load(f, Loader=yaml.FullLoader))


def load_command_sequence(config_file: str, default_height: float) -> CommandSequencer | None:
    with open(config_file, "r") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    steps_raw = raw.get("command_sequence", None)
    if steps_raw is None:
        return None
    steps = [CommandStep(**s) for s in steps_raw]
    seq = CommandSequencer(steps, default_height)
    print(f"Command sequence loaded: {len(steps)} steps, {seq.total_duration:.1f}s total.")
    return seq


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in ("true", "1", "yes", "y", "on"):
        return True
    if normalized in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def make_teleop(config: SimConfig) -> HeadlessTeleop:
    return HeadlessTeleop(
        config_init=config.cmd_init,
        max_lin=config.max_lin,
        max_ang=config.max_ang,
        height_init=config.cmd_height_init,
        height_step=config.height_step,
        min_height=config.min_height,
        max_height=config.max_height,
    )


def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded

    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0

    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0

    return a - b + c


def get_gravity_orientation(quaternion):
    """Get the gravity vector in the robot base frame."""
    q = np.array(quaternion)
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2 * (-q[1] * q[3] + q[0] * q[2])
    gravity[1] = -2 * (q[2] * q[3] + q[0] * q[1])
    gravity[2] = 1 - 2 * (q[0] * q[0] + q[3] * q[3])
    return gravity


def apply_diamond_constraint(cmd, max_lin, max_ang):
    """
    Apply an L1 velocity limit:
    |v_x| / v_max + |w_z| / w_max <= 1
    """
    limit_vx = max_lin if max_lin >= 1e-6 else 1.0
    limit_wz = max_ang if max_ang >= 1e-6 else 1.0

    ratio = abs(cmd[0]) / limit_vx + abs(cmd[2]) / limit_wz
    if ratio > 1.0:
        cmd *= 1.0 / ratio
    return cmd


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position and velocity targets."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def read_sensors(data: mujoco.MjData, config: SimConfig):
    num_actions = config.num_actions
    qpos = data.sensordata[:num_actions]
    qvel = data.sensordata[num_actions : 2 * num_actions]
    tau_meas = data.sensordata[2 * num_actions : 3 * num_actions]
    imu_quat = data.sensordata[3 * num_actions : 3 * num_actions + 4]
    ang_vel_b = data.sensordata[3 * num_actions + 4 : 3 * num_actions + 7]
    imu_acc = data.sensordata[3 * num_actions + 7 : 3 * num_actions + 10]
    lin_vel_i = data.sensordata[3 * num_actions + 13 : 3 * num_actions + 16]
    return qpos, qvel, tau_meas, imu_quat, ang_vel_b, imu_acc, lin_vel_i


def get_policy_joint_state(qpos, qvel, config: SimConfig):
    if config.policy_index_map is None:
        return qpos, qvel, config.default_angles

    return (
        qpos[config.policy_index_map],
        qvel[config.policy_index_map],
        config.default_angles[config.policy_index_map],
    )


def build_observation(
    qpos_obs,
    qvel_obs,
    default_angles_obs,
    ang_vel_b,
    gravity_b,
    cmd_vel,
    cmd_height,
    action,
    config: SimConfig,
):
    valid_leg_idx = [
        i for i in config.leg_joint_indices if i < len(qpos_obs) and i < len(default_angles_obs)
    ]
    leg_pos_delta = (qpos_obs[valid_leg_idx] - default_angles_obs[valid_leg_idx]) * config.dof_pos_scale
    leg_pos_delta = leg_pos_delta.astype(np.float32).ravel()

    obs_list = [
        ang_vel_b * config.ang_vel_scale,
        gravity_b,
        cmd_vel * config.cmd_scale,
        leg_pos_delta,
        qvel_obs * config.dof_vel_scale,
        action.astype(np.float32),
    ]
    if config.enable_height_command:
        obs_list.insert(3, cmd_height * config.height_scale)

    return obs_list


def update_observation_history(obs_history_buffer, obs_list):
    obs_tensors = [
        torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs
        for obs in obs_list
    ]
    current_obs = torch.cat(obs_tensors, dim=0)

    obs_history_buffer = torch.roll(obs_history_buffer, shifts=-1, dims=0)
    obs_history_buffer[-1] = current_obs

    split_sizes = [obs.numel() for obs in obs_tensors]
    feature_groups = torch.split(obs_history_buffer, split_sizes, dim=1)
    flat_groups = [group.flatten() for group in feature_groups]
    obs_tensor = torch.cat(flat_groups).unsqueeze(0)
    return obs_history_buffer, torch.clip(obs_tensor, -100, 100)


def apply_policy_action(action, target_dof_pos, target_dof_vel, config: SimConfig):
    for idx in config.leg_joint_indices:
        idx_xml = config.policy_index_map[idx] if config.policy_index_map is not None else idx
        if idx_xml < len(target_dof_pos) and idx < len(action):
            target_dof_pos[idx_xml] = (
                config.default_angles[idx_xml] + action[idx] * config.pos_action_scale
            )

    for idx in config.wheel_joint_indices:
        idx_xml = config.policy_index_map[idx] if config.policy_index_map is not None else idx
        if idx_xml < len(target_dof_vel) and idx < len(action):
            target_dof_vel[idx_xml] = action[idx] * config.vel_action_scale


def record_step(
    buffers: HistoryBuffers,
    lin_vel_b,
    ang_vel_b,
    gravity_b,
    qpos_obs,
    qvel_obs,
    action,
    cmd_vel,
    tau,
    counter,
    config: SimConfig,
):
    scaled_action = action.copy()
    scaled_action[config.leg_joint_indices] *= config.pos_action_scale
    scaled_action[config.wheel_joint_indices] *= config.vel_action_scale

    buffers.lin_vel.append(lin_vel_b.copy())
    buffers.ang_vel.append(ang_vel_b.copy())
    buffers.gravity_b.append(gravity_b.copy())
    buffers.joint_pos.append(qpos_obs.copy())
    buffers.joint_vel.append(qvel_obs.copy())
    buffers.action.append(scaled_action)
    buffers.time.append(counter * config.simulation_dt)
    buffers.cmd.append(cmd_vel.copy())
    buffers.tau.append(tau.copy())


def run_simulation(
    config: SimConfig,
    policy,
    teleop: HeadlessTeleop,
    enable_sipo: bool = False,
    enable_capo: bool = False,
    enable_capo_test: bool = False,
    capo_thresholds: list[float] | None = None,
    sequencer: CommandSequencer | None = None,
) -> tuple[
    HistoryBuffers,
    SipoBuffers | None,
    CapoBuffers | None,
    CapoDiagnosticRunner | None,
]:
    target_dof_pos = config.default_angles.copy()
    target_dof_vel = np.zeros(config.num_actions)
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs_history_buffer = torch.zeros((config.obs_buffer_size, config.one_step_obs_size))
    buffers = HistoryBuffers()

    model = mujoco.MjModel.from_xml_path(config.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = config.simulation_dt
    base_body_id = 1
    sipo_runner = None
    capo_runner = None
    capo_diagnostic_runner = None
    counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        viewer.sync()
        if enable_sipo:
            sipo_runner = SipoRunner.create(model, data, config, base_body_id)
        if enable_capo:
            capo_runner = CapoRunner.create(config, base_body_id)
            print("CAPO enabled: wheel sign=+1, force threshold=-15.0")
        if enable_capo_test:
            thresholds = capo_thresholds or [-10.0, -20.0, -30.0, -40.0, -60.0, -80.0]
            capo_diagnostic_runner = CapoDiagnosticRunner(
                model,
                thresholds=thresholds,
                update_rate_hz=1.0 / config.simulation_dt,
            )
            print(
                f"CAPO diagnostics: {len(capo_diagnostic_runner.candidates)} candidates "
                f"(wheel signs +/-1, thresholds={thresholds})"
            )

        while viewer.is_running() and counter * config.simulation_dt < config.simulation_duration:
            step_start = time.time()

            qpos_for_pd = data.sensordata[: config.num_actions]
            qvel_for_pd = data.sensordata[config.num_actions : 2 * config.num_actions]
            tau = pd_control(
                target_dof_pos,
                qpos_for_pd,
                config.kps,
                target_dof_vel,
                qvel_for_pd,
                config.kds,
            )
            data.ctrl[:] = tau

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xipos[base_body_id]
            counter += 1

            qpos, qvel, tau_meas, imu_quat, ang_vel_b, imu_acc, lin_vel_i = read_sensors(
                data, config
            )
            qpos_obs, qvel_obs, default_angles_obs = get_policy_joint_state(qpos, qvel, config)

            if sipo_runner is not None:
                sipo_runner.update(model, data, qpos, qvel, imu_acc, ang_vel_b, config)
            if capo_runner is not None:
                capo_runner.update(
                    data,
                    qpos,
                    qvel,
                    tau_meas,
                    imu_quat,
                    ang_vel_b,
                    imu_acc,
                    lin_vel_i,
                    round(counter * config.simulation_dt * 1000.0),
                )
            if capo_diagnostic_runner is not None:
                capo_diagnostic_runner.update(
                    model,
                    data,
                    qpos,
                    qvel,
                    tau_meas,
                    imu_quat,
                    ang_vel_b,
                    imu_acc,
                    lin_vel_i,
                    round(counter * config.simulation_dt * 1000.0),
                )

            if sequencer is not None:
                sim_time = counter * config.simulation_dt
                cmd_vel, cmd_height_val = sequencer.get_command(sim_time)
            else:
                cmd_vel = np.array(teleop.get_command(), dtype=np.float32)
                cmd_height_val = teleop.get_height_command()
            cmd_vel = apply_diamond_constraint(cmd_vel, config.max_lin, config.max_ang)
            cmd_height = np.array([cmd_height_val], dtype=np.float32)

            lin_vel_b = quat_rotate_inverse(imu_quat, lin_vel_i)
            gravity_b = get_gravity_orientation(imu_quat)

            obs_list = build_observation(
                qpos_obs,
                qvel_obs,
                default_angles_obs,
                ang_vel_b,
                gravity_b,
                cmd_vel,
                cmd_height,
                action,
                config,
            )

            record_step(
                buffers,
                lin_vel_b,
                ang_vel_b,
                gravity_b,
                qpos_obs,
                qvel_obs,
                action,
                cmd_vel,
                tau,
                counter,
                config,
            )

            if counter % config.control_decimation == 0 and counter > 0:
                obs_history_buffer, obs_tensor = update_observation_history(
                    obs_history_buffer, obs_list
                )
                action = policy(obs_tensor).detach().numpy().squeeze()
                apply_policy_action(action, target_dof_pos, target_dof_vel, config)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return (
        buffers,
        sipo_runner.buffers if sipo_runner is not None else None,
        capo_runner.buffers if capo_runner is not None else None,
        capo_diagnostic_runner,
    )


def plot_history(buffers: HistoryBuffers):
    fig_hist = plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    for i in range(3):
        plt.plot(buffers.time, [step[i] for step in buffers.lin_vel], label=f"Linear Velocity {i}")
    plt.plot(buffers.time, [step[0] for step in buffers.cmd], label="Command Velocity x", linestyle="--")
    plt.title("History Linear Velocity", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.plot(buffers.time, [step[i] for step in buffers.ang_vel], label=f"Angular Velocity {i}")
    plt.plot(buffers.time, [step[2] for step in buffers.cmd], label="Command Velocity yaw", linestyle="--")
    plt.title("History Angular Velocity", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    for i in (3, 7):
        plt.plot(buffers.time, [step[i] for step in buffers.tau], label=f"Joint Torque {i}")
    for i in (6, 7):
        plt.plot(buffers.time, [step[i] for step in buffers.joint_vel], label=f"Joint vel {i}")
    for i in (6, 7):
        plt.plot(buffers.time, [step[i] for step in buffers.action], label=f"Joint action {i}", linestyle="--")
    plt.title("History Joint", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    for i in (0, 4):
        plt.plot(buffers.time, [step[i] for step in buffers.tau], label=f"Joint Torque {i}")
    for i in (0, 1):
        plt.plot(buffers.time, [step[i] for step in buffers.joint_pos], label=f"Joint pos {i}")
    for i in (0, 1):
        plt.plot(buffers.time, [step[i] for step in buffers.action], label=f"Joint action {i}", linestyle="--")
    plt.title("History Joint", fontsize=10, pad=10)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "history_data.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig_hist)


def plot_sipo(buffers: SipoBuffers, time_data: list[float]):
    fig_sipo = plt.figure(figsize=(12, 20))
    t = time_data

    ax_traj = plt.subplot(5, 1, 1)
    ax_traj.set_title("2D Position Trajectory (XY Plane)")
    ax_traj.plot([p[0] for p in buffers.pos], [p[1] for p in buffers.pos], label="SIPO")
    ax_traj.plot(
        [p[0] for p in buffers.gt_pos],
        [p[1] for p in buffers.gt_pos],
        label="GT",
        linestyle="--",
    )
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend()
    ax_traj.grid()
    ax_traj.axis("equal")

    ax_vx = plt.subplot(5, 2, 3)
    ax_vx.set_title("Velocity X (World)")
    ax_vx.plot(t, [v[0] for v in buffers.vel], label="SIPO")
    ax_vx.plot(t, [v[0] for v in buffers.gt_vel], label="GT", linestyle="--")
    ax_vx.set_xlabel("Time (s)")
    ax_vx.legend()
    ax_vx.grid()

    ax_vy = plt.subplot(5, 2, 4)
    ax_vy.set_title("Velocity Y (World)")
    ax_vy.plot(t, [v[1] for v in buffers.vel], label="SIPO")
    ax_vy.plot(t, [v[1] for v in buffers.gt_vel], label="GT", linestyle="--")
    ax_vy.set_xlabel("Time (s)")
    ax_vy.legend()
    ax_vy.grid()

    ax_vbx = plt.subplot(5, 2, 5)
    ax_vbx.set_title("Velocity X (Body/Robot Frame)")
    ax_vbx.plot(t, [v[0] for v in buffers.vel_body], label="SIPO Body")
    ax_vbx.plot(t, [v[0] for v in buffers.gt_vel_body], label="GT Body", linestyle="--")
    ax_vbx.set_xlabel("Time (s)")
    ax_vbx.legend()
    ax_vbx.grid()

    ax_vby = plt.subplot(5, 2, 6)
    ax_vby.set_title("Velocity Y (Body/Robot Frame)")
    ax_vby.plot(t, [v[1] for v in buffers.vel_body], label="SIPO Body")
    ax_vby.plot(t, [v[1] for v in buffers.gt_vel_body], label="GT Body", linestyle="--")
    ax_vby.set_xlabel("Time (s)")
    ax_vby.legend()
    ax_vby.grid()

    ax_yaw = plt.subplot(5, 1, 4)
    ax_yaw.set_title("Yaw Angular Velocity (Z-axis) [Rad/s]")
    ax_yaw.plot(t, buffers.imu_yaw_rate, label="IMU Raw", color="purple", alpha=0.5, linewidth=1.0)
    ax_yaw.plot(t, buffers.yaw_rate, label="SIPO Est (Corrected)", color="blue", linewidth=1.5)
    ax_yaw.plot(t, buffers.gt_yaw_rate, label="GT Body", color="green", linestyle="--", linewidth=1.5)
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.legend()
    ax_yaw.grid()

    ax_z = plt.subplot(5, 1, 5)
    ax_z.set_title("Z Height (Position Z) [m]")
    ax_z.plot(t, [p[2] for p in buffers.pos], label="SIPO")
    ax_z.plot(t, [p[2] for p in buffers.gt_pos], label="GT", linestyle="--")
    ax_z.set_xlabel("Time (s)")
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "sipo_results.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig_sipo)


def plot_capo(buffers: CapoBuffers, time_data: list[float]) -> None:
    """Plot CAPO estimates using the same comparisons as the SIPO figure."""
    t = np.asarray(time_data[:len(buffers.pos)])
    pos = np.asarray(buffers.pos)
    vel_world = np.asarray(buffers.vel_world)
    vel_body = np.asarray(buffers.vel_body)
    angular_velocity = np.asarray(buffers.angular_velocity)
    gt_pos = np.asarray(buffers.gt_pos)
    gt_vel_world = np.asarray(buffers.gt_vel_world)
    gt_vel_body = np.asarray(buffers.gt_vel_body)

    fig = plt.figure(figsize=(12, 20))

    ax_traj = plt.subplot(5, 1, 1)
    ax_traj.set_title("CAPO 2D Position Trajectory (XY Plane)")
    ax_traj.plot(pos[:, 0], pos[:, 1], label="CAPO")
    ax_traj.plot(gt_pos[:, 0], gt_pos[:, 1], "--", label="GT")
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.legend()
    ax_traj.grid()
    ax_traj.axis("equal")

    ax_vx = plt.subplot(5, 2, 3)
    ax_vx.set_title("Velocity X (World)")
    ax_vx.plot(t, vel_world[:, 0], label="CAPO")
    ax_vx.plot(t, gt_vel_world[:, 0], "--", label="GT")
    ax_vx.legend()
    ax_vx.grid()

    ax_vy = plt.subplot(5, 2, 4)
    ax_vy.set_title("Velocity Y (World)")
    ax_vy.plot(t, vel_world[:, 1], label="CAPO")
    ax_vy.plot(t, gt_vel_world[:, 1], "--", label="GT")
    ax_vy.legend()
    ax_vy.grid()

    ax_vbx = plt.subplot(5, 2, 5)
    ax_vbx.set_title("Velocity X (Body/Robot Frame)")
    ax_vbx.plot(t, vel_body[:, 0], label="CAPO Body")
    ax_vbx.plot(t, gt_vel_body[:, 0], "--", label="GT Body")
    ax_vbx.legend()
    ax_vbx.grid()

    ax_vby = plt.subplot(5, 2, 6)
    ax_vby.set_title("Velocity Y (Body/Robot Frame)")
    ax_vby.plot(t, vel_body[:, 1], label="CAPO Body")
    ax_vby.plot(t, gt_vel_body[:, 1], "--", label="GT Body")
    ax_vby.legend()
    ax_vby.grid()

    ax_yaw = plt.subplot(5, 1, 4)
    ax_yaw.set_title("Yaw Angular Velocity [rad/s]")
    ax_yaw.plot(t, buffers.imu_yaw_rate, label="IMU raw", alpha=0.5)
    ax_yaw.plot(t, angular_velocity[:, 2], label="CAPO")
    ax_yaw.plot(t, buffers.gt_yaw_rate, "--", label="GT body")
    ax_yaw.legend()
    ax_yaw.grid()

    ax_z = plt.subplot(5, 1, 5)
    ax_z.set_title("Z Height [m]")
    ax_z.plot(t, pos[:, 2], label="CAPO")
    ax_z.plot(t, gt_pos[:, 2], "--", label="GT")
    ax_z.set_xlabel("Time [s]")
    ax_z.legend()
    ax_z.grid()

    fig.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOG_DIR / "capo_results.png", dpi=300)
    plt.show()
    plt.close(fig)


def _capo_candidate_metrics(buffers: CapoCandidateBuffers) -> dict[str, float]:
    est_vel = np.asarray(buffers.vel_body)
    gt_vel = np.asarray(buffers.gt_vel_body)
    contact_est = np.asarray(buffers.contact_est, dtype=bool).reshape(-1)
    contact_gt = np.asarray(buffers.contact_gt, dtype=bool).reshape(-1)

    error = est_vel - gt_vel
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    moving = np.abs(gt_vel[:, 0]) > 0.1
    moving_vx_rmse = (
        float(np.sqrt(np.mean(np.square(error[moving, 0]))))
        if np.any(moving)
        else float(rmse[0])
    )

    tp = int(np.sum(contact_est & contact_gt))
    fp = int(np.sum(contact_est & ~contact_gt))
    fn = int(np.sum(~contact_est & contact_gt))
    tn = int(np.sum(~contact_est & ~contact_gt))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)

    return {
        "rmse_x": float(rmse[0]),
        "rmse_y": float(rmse[1]),
        "rmse_z": float(rmse[2]),
        "moving_vx_rmse": moving_vx_rmse,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def report_and_plot_capo(runner: CapoDiagnosticRunner, time_data: list[float]) -> None:
    """Print candidate scores and plot the recommended CAPO configuration."""
    scored = [
        (candidate, _capo_candidate_metrics(candidate.buffers))
        for candidate in runner.candidates
    ]
    print("\nCAPO wheel-sign / threshold diagnostics")
    print(
        " sign threshold | vx_RMSE moving_vx | contact_F1 precision recall "
        "| TP FP FN TN"
    )
    for candidate, metrics in scored:
        b = candidate.buffers
        print(
            f" {b.wheel_sign:+d} {b.threshold:9.2f} |"
            f" {metrics['rmse_x']:7.4f} {metrics['moving_vx_rmse']:9.4f} |"
            f" {metrics['f1']:10.3f} {metrics['precision']:9.3f}"
            f" {metrics['recall']:6.3f} |"
            f" {int(metrics['tp']):4d} {int(metrics['fp']):4d}"
            f" {int(metrics['fn']):4d} {int(metrics['tn']):4d}"
        )

    # Force/contact classification is independent of wheel encoder sign, so
    # use the +1 candidates to select threshold. Break F1 ties by accuracy.
    positive_sign = [(c, m) for c, m in scored if c.buffers.wheel_sign == 1]
    best_threshold_candidate, best_threshold_metrics = max(
        positive_sign,
        key=lambda item: (item[1]["f1"], item[1]["accuracy"]),
    )
    best_threshold = best_threshold_candidate.buffers.threshold

    same_threshold = [
        (c, m) for c, m in scored if c.buffers.threshold == best_threshold
    ]
    best_candidate, best_metrics = min(
        same_threshold, key=lambda item: item[1]["moving_vx_rmse"]
    )

    force_z = np.asarray(best_candidate.buffers.force_z).reshape(-1)
    contact_gt = np.asarray(best_candidate.buffers.contact_gt, dtype=bool).reshape(-1)
    print(
        f"\nRecommended threshold: {best_threshold:.2f} "
        f"(contact F1={best_threshold_metrics['f1']:.3f})"
    )
    print(
        f"Recommended wheel sign: {best_candidate.buffers.wheel_sign:+d} "
        f"(moving vx RMSE={best_metrics['moving_vx_rmse']:.4f} m/s)"
    )
    if np.any(contact_gt):
        print(
            "Force Z while MuJoCo contact [p05, p50, p95]: "
            f"{np.percentile(force_z[contact_gt], [5, 50, 95])}"
        )
    if np.any(~contact_gt):
        print(
            "Force Z while MuJoCo no-contact [p05, p50, p95]: "
            f"{np.percentile(force_z[~contact_gt], [5, 50, 95])}"
        )
    else:
        print(
            "Warning: no airborne wheel samples were recorded; threshold "
            "false-positive performance cannot be determined."
        )

    b = best_candidate.buffers
    t = np.asarray(time_data[:len(b.vel_body)])
    vel = np.asarray(b.vel_body)
    gt_vel = np.asarray(b.gt_vel_body)
    forces = np.asarray(b.force_z)
    contact_gt_2d = np.asarray(b.contact_gt, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(t, gt_vel[:, 0], "k--", label="GT body vx")
    axes[0].plot(t, vel[:, 0], label="CAPO body vx")
    axes[0].set_ylabel("Velocity [m/s]")
    axes[0].legend()
    axes[0].grid()

    for wheel in range(2):
        label = "left" if wheel == 0 else "right"
        axes[1].plot(t, forces[:, wheel], label=f"{label} estimated Fz")
    axes[1].axhline(best_threshold, color="red", linestyle="--", label="threshold")
    axes[1].set_ylabel("Estimated force Z")
    axes[1].legend()
    axes[1].grid()

    axes[2].step(t, contact_gt_2d[:, 0], where="post", label="GT left contact")
    axes[2].step(t, contact_gt_2d[:, 1], where="post", label="GT right contact")
    axes[2].set_ylabel("Contact")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid()

    fig.suptitle(
        f"CAPO diagnostics: sign={b.wheel_sign:+d}, threshold={best_threshold:.2f}"
    )
    fig.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOG_DIR / "capo_diagnostics.png", dpi=300)
    plt.show()
    plt.close(fig)


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the yaml config file",
    )
    parser.add_argument(
        "--sipo",
        type=parse_bool,
        default=False,
        help="enable SIPO estimation and result plotting (true/false)",
    )
    parser.add_argument(
        "--capo",
        type=parse_bool,
        default=False,
        help="enable fixed CAPO estimation and SIPO-style result plotting",
    )
    parser.add_argument(
        "--seq",
        type=parse_bool,
        default=False,
        help="run scripted command sequence from config instead of teleop (true/false)",
    )
    parser.add_argument(
        "--capo-test",
        type=parse_bool,
        default=False,
        help="test CAPO wheel sign and contact-force thresholds (true/false)",
    )
    parser.add_argument(
        "--capo-thresholds",
        type=parse_thresholds,
        default=[-10.0, -20.0, -30.0, -40.0, -60.0, -80.0],
        help="comma-separated CAPO vertical-force thresholds",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sequencer = load_command_sequence(args.config, config.cmd_height_init) if args.seq else None
    policy = torch.jit.load(config.policy_path)
    teleop = make_teleop(config)
    if sequencer is None:
        print("Headless teleop active.")
    else:
        print("Running scripted command sequence (keyboard/gamepad ignored).")

    try:
        buffers, sipo_buffers, capo_buffers, capo_diagnostic_runner = run_simulation(
            config,
            policy,
            teleop,
            enable_sipo=args.sipo,
            enable_capo=args.capo,
            enable_capo_test=args.capo_test,
            capo_thresholds=args.capo_thresholds,
            sequencer=sequencer,
        )
        if sipo_buffers is not None:
            plot_sipo(sipo_buffers, buffers.time)
        if capo_buffers is not None:
            plot_capo(capo_buffers, buffers.time)
        if capo_diagnostic_runner is not None:
            report_and_plot_capo(capo_diagnostic_runner, buffers.time)
        plot_history(buffers)
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
