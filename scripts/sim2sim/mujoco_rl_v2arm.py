import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from headless_teleop import HeadlessTeleop


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "pineapple_v2_5_arm.yaml"
LOG_DIR = Path(__file__).resolve().parent / "logs"


DEFAULT_LOCO_JOINT_NAMES = [
    "L_hip_joint",
    "L_thigh_joint",
    "L_calf_joint",
    "L_wheel_joint",
    "R_hip_joint",
    "R_thigh_joint",
    "R_calf_joint",
    "R_wheel_joint",
]
DEFAULT_LEG_JOINT_NAMES = [
    "L_hip_joint",
    "L_thigh_joint",
    "L_calf_joint",
    "R_hip_joint",
    "R_thigh_joint",
    "R_calf_joint",
]
DEFAULT_WHEEL_JOINT_NAMES = ["L_wheel_joint", "R_wheel_joint"]
DEFAULT_ARM_JOINT_NAMES = [
    "arm_base_joint",
    "arm_link1_joint",
    "arm_link2_joint",
    "arm_link4_joint",
    "arm_link5_joint",
    "arm_link6_joint",
]
DEFAULT_LOCO_ACTUATOR_NAMES = [
    "L_hip",
    "L_thigh",
    "L_calf",
    "L_wheel",
    "R_hip",
    "R_thigh",
    "R_calf",
    "R_wheel",
]
DEFAULT_ARM_ACTUATOR_NAMES = [
    "arm_actuator1",
    "arm_actuator2",
    "arm_actuator3",
    "arm_actuator4",
    "arm_actuator5",
    "arm_actuator6",
]


@dataclass
class SimConfig:
    policy_path: str
    xml_path: str
    simulation_duration: float
    simulation_dt: float
    control_decimation: int
    kps: np.ndarray
    kds: np.ndarray
    arm_kps: np.ndarray
    arm_kds: np.ndarray
    default_loco_angles: np.ndarray
    default_arm_angles: np.ndarray
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    pos_action_scale: float
    vel_action_scale: float
    arm_pos_action_scale: float
    cmd_scale: np.ndarray
    num_actions: int
    num_obs: int
    one_step_obs_size: int
    obs_buffer_size: int
    loco_joint_names: list[str]
    leg_joint_names: list[str]
    wheel_joint_names: list[str]
    arm_joint_names: list[str]
    policy_joint_pos_names: list[str]
    policy_joint_vel_names: list[str]
    loco_actuator_names: list[str]
    arm_actuator_names: list[str]
    cmd_init: np.ndarray
    ee_pose_init: np.ndarray
    ee_pos_min: np.ndarray
    ee_pos_max: np.ndarray
    ee_pos_step: float
    ee_rot_step: float
    max_lin: float
    max_ang: float

    @classmethod
    def from_dict(cls, config: dict) -> "SimConfig":
        loco_joint_names = config.get("loco_joint_names", DEFAULT_LOCO_JOINT_NAMES)
        leg_joint_names = config.get("leg_joint_names", DEFAULT_LEG_JOINT_NAMES)
        wheel_joint_names = config.get("wheel_joint_names", DEFAULT_WHEEL_JOINT_NAMES)
        arm_joint_names = config.get("arm_joint_names", DEFAULT_ARM_JOINT_NAMES)
        policy_joint_pos_names = config.get("policy_joint_pos_names", leg_joint_names + arm_joint_names)
        policy_joint_vel_names = config.get(
            "policy_joint_vel_names",
            leg_joint_names + wheel_joint_names + arm_joint_names,
        )
        loco_actuator_names = config.get("loco_actuator_names", DEFAULT_LOCO_ACTUATOR_NAMES)
        arm_actuator_names = config.get("arm_actuator_names", DEFAULT_ARM_ACTUATOR_NAMES)

        sim_config = cls(
            policy_path=config["policy_path"],
            xml_path=config["xml_path"],
            simulation_duration=config["simulation_duration"],
            simulation_dt=config["simulation_dt"],
            control_decimation=config["control_decimation"],
            kps=np.array(config["kps"], dtype=np.float32),
            kds=np.array(config["kds"], dtype=np.float32),
            arm_kps=np.array(config["arm_kps"], dtype=np.float32),
            arm_kds=np.array(config["arm_kds"], dtype=np.float32),
            default_loco_angles=np.array(config["default_loco_angles"], dtype=np.float32),
            default_arm_angles=np.array(config["default_arm_angles"], dtype=np.float32),
            lin_vel_scale=config["lin_vel_scale"],
            ang_vel_scale=config["ang_vel_scale"],
            dof_pos_scale=config["dof_pos_scale"],
            dof_vel_scale=config["dof_vel_scale"],
            pos_action_scale=config["pos_action_scale"],
            vel_action_scale=config["vel_action_scale"],
            arm_pos_action_scale=config["arm_pos_action_scale"],
            cmd_scale=np.array(config["cmd_scale"], dtype=np.float32),
            num_actions=config["num_actions"],
            num_obs=config["num_obs"],
            one_step_obs_size=config["one_step_obs_size"],
            obs_buffer_size=config["obs_buffer_size"],
            loco_joint_names=loco_joint_names,
            leg_joint_names=leg_joint_names,
            wheel_joint_names=wheel_joint_names,
            arm_joint_names=arm_joint_names,
            policy_joint_pos_names=policy_joint_pos_names,
            policy_joint_vel_names=policy_joint_vel_names,
            loco_actuator_names=loco_actuator_names,
            arm_actuator_names=arm_actuator_names,
            cmd_init=np.array(config["cmd_init"], dtype=np.float32),
            ee_pose_init=np.array(config["ee_pose_init"], dtype=np.float32),
            ee_pos_min=np.array(config.get("ee_pos_min", [0.3, -0.3, 0.0]), dtype=np.float32),
            ee_pos_max=np.array(config.get("ee_pos_max", [0.7, 0.3, 0.4]), dtype=np.float32),
            ee_pos_step=config.get("ee_pos_step", 0.01),
            ee_rot_step=config.get("ee_rot_step", 0.05),
            max_lin=config.get("max_lin_vel", 1.0),
            max_ang=config.get("max_ang_vel", 1.0),
        )
        sim_config.validate()
        return sim_config

    def validate(self):
        expected_actions = len(self.leg_joint_names) + len(self.wheel_joint_names) + len(self.arm_joint_names)
        expected_one_step = 3 + 3 + 3 + 7 + len(self.policy_joint_pos_names)
        expected_one_step += len(self.policy_joint_vel_names)
        expected_one_step += expected_actions
        joint_names = set(self.loco_joint_names + self.arm_joint_names)

        checks = [
            (len(self.kps), len(self.loco_joint_names), "kps"),
            (len(self.kds), len(self.loco_joint_names), "kds"),
            (len(self.default_loco_angles), len(self.loco_joint_names), "default_loco_angles"),
            (len(self.arm_kps), len(self.arm_joint_names), "arm_kps"),
            (len(self.arm_kds), len(self.arm_joint_names), "arm_kds"),
            (len(self.default_arm_angles), len(self.arm_joint_names), "default_arm_angles"),
            (set(self.policy_joint_pos_names).issubset(joint_names), True, "policy_joint_pos_names"),
            (set(self.policy_joint_vel_names).issubset(joint_names), True, "policy_joint_vel_names"),
            (len(self.loco_actuator_names), len(self.loco_joint_names), "loco_actuator_names"),
            (len(self.arm_actuator_names), len(self.arm_joint_names), "arm_actuator_names"),
            (len(self.cmd_scale), 3, "cmd_scale"),
            (len(self.ee_pose_init), 7, "ee_pose_init"),
            (len(self.ee_pos_min), 3, "ee_pos_min"),
            (len(self.ee_pos_max), 3, "ee_pos_max"),
            (self.num_actions, expected_actions, "num_actions"),
            (self.one_step_obs_size, expected_one_step, "one_step_obs_size"),
            (self.num_obs, self.one_step_obs_size * self.obs_buffer_size, "num_obs"),
        ]
        for actual, expected, field_name in checks:
            if actual != expected:
                raise ValueError(f"{field_name} must be {expected}, got {actual}.")


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


def load_config(config_file: str) -> SimConfig:
    with open(config_file, "r") as f:
        return SimConfig.from_dict(yaml.load(f, Loader=yaml.FullLoader))


def make_teleop(config: SimConfig) -> HeadlessTeleop:
    return HeadlessTeleop(
        config_init=config.cmd_init,
        max_lin=config.max_lin,
        max_ang=config.max_ang,
        ee_pose_init=config.ee_pose_init,
        ee_pos_step=config.ee_pos_step,
        ee_rot_step=config.ee_rot_step,
        ee_pos_min=config.ee_pos_min,
        ee_pos_max=config.ee_pos_max,
    )


def joint_sensor_prefix(joint_name: str) -> str:
    return joint_name.removesuffix("_joint")


def read_sensor(data: mujoco.MjData, model: mujoco.MjModel, sensor_name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo sensor '{sensor_name}' was not found in the XML.")

    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()


def read_joint_sensors(data: mujoco.MjData, model: mujoco.MjModel, joint_names: list[str], suffix: str) -> np.ndarray:
    values = [read_sensor(data, model, f"{joint_sensor_prefix(joint_name)}_{suffix}") for joint_name in joint_names]
    return np.concatenate(values).astype(np.float32)


def set_actuator_controls(model: mujoco.MjModel, ctrl: np.ndarray, actuator_names: list[str], values: np.ndarray):
    for actuator_name, value in zip(actuator_names, values):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator '{actuator_name}' was not found in the XML.")
        ctrl[actuator_id] = value


def validate_model_bindings(model: mujoco.MjModel, config: SimConfig):
    print("\n=== MuJoCo arm binding check ===")
    for idx, (actuator_name, joint_name) in enumerate(zip(config.arm_actuator_names, config.arm_joint_names)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint '{joint_name}' was not found in the XML.")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator '{actuator_name}' was not found in the XML.")

        actuator_joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_joint_id)
        if actuator_joint_name != joint_name:
            raise ValueError(
                f"Arm binding mismatch at index {idx}: config maps '{actuator_name}' to '{joint_name}', "
                f"but XML maps it to '{actuator_joint_name}'."
            )

        qpos_addr = int(model.jnt_qposadr[joint_id])
        key_qpos = float(model.key_qpos[0, qpos_addr]) if model.nkey > 0 else None
        default_qpos = float(config.default_arm_angles[idx])
        print(
            f"  arm[{idx}] {actuator_name:14s} -> {joint_name:18s} "
            f"joint_id={joint_id:2d} qposadr={qpos_addr:2d} key={key_qpos: .6f} cfg_default={default_qpos: .6f}"
        )

        if key_qpos is not None and abs(key_qpos - default_qpos) > 1e-4:
            print(
                f"  [WARN] keyframe/default mismatch for {joint_name}: "
                f"key={key_qpos: .6f}, cfg_default={default_qpos: .6f}"
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


def normalize_quat(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_to_mat(q):
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def transform_ee_pose_to_world(ee_pose_b, base_pos_w, base_quat_w):
    base_rot_w = quat_to_mat(base_quat_w)
    ee_pos_w = base_pos_w + base_rot_w @ ee_pose_b[:3]
    ee_quat_w = quat_mul(base_quat_w, ee_pose_b[3:])
    return ee_pos_w.astype(np.float32), normalize_quat(ee_quat_w)


def add_sphere_marker(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos.astype(np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )


def add_capsule_marker(scene, start, end, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        start.astype(np.float64),
        end.astype(np.float64),
    )
    geom.rgba[:] = np.array(rgba, dtype=np.float32)


def draw_desired_ee_pose(viewer, data, base_body_id, ee_pose_b):
    base_pos_w = data.xpos[base_body_id].copy()
    base_quat_w = data.xquat[base_body_id].copy()
    ee_pos_w, ee_quat_w = transform_ee_pose_to_world(ee_pose_b, base_pos_w, base_quat_w)
    ee_rot_w = quat_to_mat(ee_quat_w)

    scene = viewer.user_scn
    scene.ngeom = 0
    add_sphere_marker(scene, ee_pos_w, 0.025, [1.0, 0.85, 0.0, 0.55])

    axis_len = 0.12
    axis_radius = 0.006
    axis_colors = (
        [1.0, 0.1, 0.1, 0.9],
        [0.1, 0.9, 0.1, 0.9],
        [0.1, 0.35, 1.0, 0.9],
    )
    for axis_idx, color in enumerate(axis_colors):
        axis_end = ee_pos_w + ee_rot_w[:, axis_idx] * axis_len
        add_capsule_marker(scene, ee_pos_w, axis_end, axis_radius, color)


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


def read_base_sensors(data: mujoco.MjData, model: mujoco.MjModel):
    imu_quat = read_sensor(data, model, "imu_quat")
    ang_vel_b = read_sensor(data, model, "imu_gyro")
    lin_vel_i = read_sensor(data, model, "frame_lin_vel")
    return imu_quat, ang_vel_b, lin_vel_i


def values_by_name(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}


def get_policy_joint_state(loco_qpos, loco_qvel, arm_qpos, arm_qvel, config: SimConfig):
    pos_by_name = values_by_name(config.loco_joint_names, loco_qpos)
    pos_by_name.update(values_by_name(config.arm_joint_names, arm_qpos))

    vel_by_name = values_by_name(config.loco_joint_names, loco_qvel)
    vel_by_name.update(values_by_name(config.arm_joint_names, arm_qvel))

    default_by_name = values_by_name(config.loco_joint_names, config.default_loco_angles)
    default_by_name.update(values_by_name(config.arm_joint_names, config.default_arm_angles))

    return (
        np.array([pos_by_name[name] for name in config.policy_joint_pos_names], dtype=np.float32),
        np.array([vel_by_name[name] for name in config.policy_joint_vel_names], dtype=np.float32),
        np.array([default_by_name[name] for name in config.policy_joint_pos_names], dtype=np.float32),
    )


def build_observation(
    qpos_obs,
    qvel_obs,
    default_angles_obs,
    ang_vel_b,
    gravity_b,
    cmd_vel,
    ee_pose_command,
    action,
    config: SimConfig,
):
    joint_pos_delta = (qpos_obs - default_angles_obs) * config.dof_pos_scale
    joint_pos_delta = joint_pos_delta.astype(np.float32).ravel()

    obs_list = [
        ang_vel_b * config.ang_vel_scale,
        gravity_b,
        cmd_vel * config.cmd_scale,
        ee_pose_command,
        joint_pos_delta,
        qvel_obs * config.dof_vel_scale,
        action.astype(np.float32),
    ]

    return obs_list


def update_observation_history(obs_history_buffer, obs_list):
    obs_tensors = [
        torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs
        for obs in obs_list
    ]
    current_obs = torch.cat(obs_tensors, dim=0)

    obs_history_buffer = torch.roll(obs_history_buffer, shifts=-1, dims=0)
    obs_history_buffer[-1] = current_obs
    if not torch.any(obs_history_buffer[:-1]):
        obs_history_buffer[:] = current_obs

    # local_rsl_rl_go2arm_style reorders IsaacLab's per-term history into
    # timestep-major order with the newest timestep first before policy inference.
    obs_tensor = torch.flip(obs_history_buffer, dims=[0]).flatten().unsqueeze(0)
    return obs_history_buffer, torch.clip(obs_tensor, -100, 100)


def apply_policy_action(action, target_loco_pos, target_loco_vel, target_arm_pos, config: SimConfig):
    loco_default_by_name = values_by_name(config.loco_joint_names, config.default_loco_angles)
    arm_default_by_name = values_by_name(config.arm_joint_names, config.default_arm_angles)
    loco_index_by_name = {name: idx for idx, name in enumerate(config.loco_joint_names)}
    arm_index_by_name = {name: idx for idx, name in enumerate(config.arm_joint_names)}

    action_idx = 0
    for joint_name in config.leg_joint_names:
        loco_idx = loco_index_by_name[joint_name]
        target_loco_pos[loco_idx] = loco_default_by_name[joint_name] + action[action_idx] * config.pos_action_scale
        action_idx += 1

    for joint_name in config.wheel_joint_names:
        loco_idx = loco_index_by_name[joint_name]
        target_loco_vel[loco_idx] = action[action_idx] * config.vel_action_scale
        action_idx += 1

    for joint_name in config.arm_joint_names:
        arm_idx = arm_index_by_name[joint_name]
        target_arm_pos[arm_idx] = arm_default_by_name[joint_name] + action[action_idx] * config.arm_pos_action_scale
        action_idx += 1


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
    leg_end = len(config.leg_joint_names)
    wheel_end = leg_end + len(config.wheel_joint_names)
    scaled_action[:leg_end] *= config.pos_action_scale
    scaled_action[leg_end:wheel_end] *= config.vel_action_scale
    scaled_action[wheel_end:] *= config.arm_pos_action_scale

    buffers.lin_vel.append(lin_vel_b.copy())
    buffers.ang_vel.append(ang_vel_b.copy())
    buffers.gravity_b.append(gravity_b.copy())
    buffers.joint_pos.append(qpos_obs.copy())
    buffers.joint_vel.append(qvel_obs.copy())
    buffers.action.append(scaled_action)
    buffers.time.append(counter * config.simulation_dt)
    buffers.cmd.append(cmd_vel.copy())
    buffers.tau.append(tau.copy())


def compute_control_tau(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    target_loco_pos,
    target_loco_vel,
    target_arm_pos,
    target_arm_vel,
    config: SimConfig,
):
    loco_qpos = read_joint_sensors(data, model, config.loco_joint_names, "pos")
    loco_qvel = read_joint_sensors(data, model, config.loco_joint_names, "vel")
    arm_qpos = read_joint_sensors(data, model, config.arm_joint_names, "pos")
    arm_qvel = read_joint_sensors(data, model, config.arm_joint_names, "vel")

    loco_tau = pd_control(
        target_loco_pos,
        loco_qpos,
        config.kps,
        target_loco_vel,
        loco_qvel,
        config.kds,
    )
    arm_tau = pd_control(
        target_arm_pos,
        arm_qpos,
        config.arm_kps,
        target_arm_vel,
        arm_qvel,
        config.arm_kds,
    )
    tau = np.zeros(model.nu, dtype=np.float32)
    set_actuator_controls(model, tau, config.loco_actuator_names, loco_tau)
    set_actuator_controls(model, tau, config.arm_actuator_names, arm_tau)
    return tau


def action_joint_names(config: SimConfig) -> list[str]:
    return config.leg_joint_names + config.wheel_joint_names + config.arm_joint_names


def print_named_vector(title: str, names: list[str], values: np.ndarray):
    print(title)
    for name, value in zip(names, values):
        print(f"  {name:18s} {value: .6f}")


def print_lr_check(label: str, names: list[str], values: np.ndarray):
    value_by_name = values_by_name(names, values)
    pairs = [
        ("hip", "L_hip_joint", "R_hip_joint"),
        ("thigh", "L_thigh_joint", "R_thigh_joint"),
        ("calf", "L_calf_joint", "R_calf_joint"),
        ("wheel", "L_wheel_joint", "R_wheel_joint"),
    ]
    print(label)
    for pair_name, left, right in pairs:
        if left in value_by_name and right in value_by_name:
            l_val = value_by_name[left]
            r_val = value_by_name[right]
            print(f"  {pair_name:6s} L={l_val: .6f} R={r_val: .6f} diff={l_val - r_val: .6f}")


def clip_actuator_ctrl(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    clipped = ctrl.copy()
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            clipped[i] = np.clip(clipped[i], model.actuator_ctrlrange[i, 0], model.actuator_ctrlrange[i, 1])
    return clipped


def run_headless_debug(config: SimConfig, policy, teleop: HeadlessTeleop, control_steps: int):
    target_loco_pos = config.default_loco_angles.copy()
    target_loco_vel = np.zeros(len(config.loco_joint_names), dtype=np.float32)
    target_arm_pos = config.default_arm_angles.copy()
    target_arm_vel = np.zeros(len(config.arm_joint_names), dtype=np.float32)
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs_history_buffer = torch.zeros((config.obs_buffer_size, config.one_step_obs_size))

    model = mujoco.MjModel.from_xml_path(config.xml_path)
    data = mujoco.MjData(model)
    validate_model_bindings(model, config)
    model.opt.timestep = config.simulation_dt
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    act_names = action_joint_names(config)
    tau_names = config.loco_joint_names + config.arm_joint_names
    tau_history = []
    applied_force_history = []
    action_history = []

    print(f"Headless debug: {control_steps} policy updates, dt={config.simulation_dt}, decimation={config.control_decimation}")
    print_named_vector("Initial target_loco_pos", config.loco_joint_names, target_loco_pos)
    print_named_vector("Initial target_arm_pos", config.arm_joint_names, target_arm_pos)

    counter = 0
    updates = 0
    while updates < control_steps:
        tau = compute_control_tau(
            data,
            model,
            target_loco_pos,
            target_loco_vel,
            target_arm_pos,
            target_arm_vel,
            config,
        )
        data.ctrl[:] = tau[:]
        tau_history.append(tau.copy())

        mujoco.mj_step(model, data)
        applied_force_history.append(data.actuator_force.copy())
        counter += 1

        loco_qpos = read_joint_sensors(data, model, config.loco_joint_names, "pos")
        loco_qvel = read_joint_sensors(data, model, config.loco_joint_names, "vel")
        arm_qpos = read_joint_sensors(data, model, config.arm_joint_names, "pos")
        arm_qvel = read_joint_sensors(data, model, config.arm_joint_names, "vel")
        imu_quat, ang_vel_b, lin_vel_i = read_base_sensors(data, model)
        qpos_obs, qvel_obs, default_angles_obs = get_policy_joint_state(
            loco_qpos, loco_qvel, arm_qpos, arm_qvel, config
        )

        cmd_vel = np.array(teleop.get_command(), dtype=np.float32)
        ee_pose_command = teleop.get_ee_pose_command()
        lin_vel_b = quat_rotate_inverse(imu_quat, lin_vel_i)
        gravity_b = get_gravity_orientation(imu_quat)

        obs_list = build_observation(
            qpos_obs,
            qvel_obs,
            default_angles_obs,
            ang_vel_b,
            gravity_b,
            cmd_vel,
            ee_pose_command,
            action,
            config,
        )

        if counter % config.control_decimation == 0:
            obs_history_buffer, obs_tensor = update_observation_history(
                obs_history_buffer, obs_list
            )
            action = policy(obs_tensor).detach().numpy().squeeze().astype(np.float32)
            apply_policy_action(action, target_loco_pos, target_loco_vel, target_arm_pos, config)
            action_history.append(action.copy())

            next_tau = compute_control_tau(
                data,
                model,
                target_loco_pos,
                target_loco_vel,
                target_arm_pos,
                target_arm_vel,
                config,
            )
            scaled_action = action.copy()
            leg_end = len(config.leg_joint_names)
            wheel_end = leg_end + len(config.wheel_joint_names)
            scaled_action[:leg_end] *= config.pos_action_scale
            scaled_action[leg_end:wheel_end] *= config.vel_action_scale
            scaled_action[wheel_end:] *= config.arm_pos_action_scale

            print(f"\n=== policy update {updates + 1} at sim step {counter}, t={counter * config.simulation_dt:.4f}s ===")
            print_named_vector("raw action", act_names, action)
            print_named_vector("scaled action target delta/vel", act_names, scaled_action)
            print_named_vector("next tau command from current state", tau_names, next_tau)
            print_named_vector("next tau clipped by XML ctrlrange", tau_names, clip_actuator_ctrl(model, next_tau))
            print_named_vector("actual actuator force from previous sim step", tau_names, applied_force_history[-1])
            print_lr_check("raw action L/R check", act_names, action)
            print_lr_check("next tau command L/R check", tau_names, next_tau)
            print(f"max |raw action|={np.max(np.abs(action)):.6f}, max |scaled action|={np.max(np.abs(scaled_action)):.6f}, max |next tau cmd|={np.max(np.abs(next_tau)):.6f}, max |next tau clipped|={np.max(np.abs(clip_actuator_ctrl(model, next_tau))):.6f}")
            updates += 1

    tau_array = np.array(tau_history)
    applied_force_array = np.array(applied_force_history)
    action_array = np.array(action_history)
    print("\n=== summary ===")
    print_named_vector("max |tau command| over stepped frames", tau_names, np.max(np.abs(tau_array), axis=0))
    print_named_vector("max |actual actuator force| over stepped frames", tau_names, np.max(np.abs(applied_force_array), axis=0))
    print_named_vector("max |raw action| over policy updates", act_names, np.max(np.abs(action_array), axis=0))
    print_lr_check("summary max |raw action| L/R check", act_names, np.max(np.abs(action_array), axis=0))
    print_lr_check("summary max |tau command| L/R check", tau_names, np.max(np.abs(tau_array), axis=0))
    print_lr_check("summary max |actual actuator force| L/R check", tau_names, np.max(np.abs(applied_force_array), axis=0))


def run_simulation(config: SimConfig, policy, teleop: HeadlessTeleop) -> HistoryBuffers:
    target_loco_pos = config.default_loco_angles.copy()
    target_loco_vel = np.zeros(len(config.loco_joint_names), dtype=np.float32)
    target_arm_pos = config.default_arm_angles.copy()
    target_arm_vel = np.zeros(len(config.arm_joint_names), dtype=np.float32)
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs_history_buffer = torch.zeros((config.obs_buffer_size, config.one_step_obs_size))
    buffers = HistoryBuffers()

    model = mujoco.MjModel.from_xml_path(config.xml_path)
    data = mujoco.MjData(model)
    validate_model_bindings(model, config)
    model.opt.timestep = config.simulation_dt
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        viewer.sync()

        while viewer.is_running() and counter * config.simulation_dt < config.simulation_duration:
            step_start = time.time()

            tau = compute_control_tau(
                data,
                model,
                target_loco_pos,
                target_loco_vel,
                target_arm_pos,
                target_arm_vel,
                config,
            )
            data.ctrl[:] = tau[:]

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xipos[base_body_id]
            counter += 1

            loco_qpos = read_joint_sensors(data, model, config.loco_joint_names, "pos")
            loco_qvel = read_joint_sensors(data, model, config.loco_joint_names, "vel")
            arm_qpos = read_joint_sensors(data, model, config.arm_joint_names, "pos")
            arm_qvel = read_joint_sensors(data, model, config.arm_joint_names, "vel")
            imu_quat, ang_vel_b, lin_vel_i = read_base_sensors(data, model)
            qpos_obs, qvel_obs, default_angles_obs = get_policy_joint_state(
                loco_qpos, loco_qvel, arm_qpos, arm_qvel, config
            )

            cmd_vel = np.array(teleop.get_command(), dtype=np.float32)
            ee_pose_command = teleop.get_ee_pose_command()
            draw_desired_ee_pose(viewer, data, base_body_id, ee_pose_command)

            lin_vel_b = quat_rotate_inverse(imu_quat, lin_vel_i)
            gravity_b = get_gravity_orientation(imu_quat)

            obs_list = build_observation(
                qpos_obs,
                qvel_obs,
                default_angles_obs,
                ang_vel_b,
                gravity_b,
                cmd_vel,
                ee_pose_command,
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
                apply_policy_action(action, target_loco_pos, target_loco_vel, target_arm_pos, config)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return buffers


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
    output_path = LOG_DIR / "history_data_v2arm.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig_hist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the yaml config file",
    )
    parser.add_argument(
        "--headless-debug-steps",
        type=int,
        default=0,
        help="run without viewer and print the first N policy updates for action/torque debugging",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    policy = torch.jit.load(config.policy_path)
    teleop = make_teleop(config)
    print("Headless teleop initialized (no GUI window).")

    try:
        if args.headless_debug_steps > 0:
            run_headless_debug(config, policy, teleop, args.headless_debug_steps)
        else:
            buffers = run_simulation(config, policy, teleop)
            plot_history(buffers)
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
