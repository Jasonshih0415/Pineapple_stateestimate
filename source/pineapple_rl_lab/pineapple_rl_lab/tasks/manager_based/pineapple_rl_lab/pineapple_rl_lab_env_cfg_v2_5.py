import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import pineapple_rl_lab.tasks.manager_based.pineapple_rl_lab.mdp as pineapple_mdp
import pineapple_rl_lab.tasks.manager_based.pineapple_rl_lab.mdp_go2arm_style as go2arm_mdp
from pineapple_rl_lab.assets.assets import ISAACLAB_ASSETS_DATA_DIR
from pineapple_rl_lab.assets.utils.pace_actuator_cfg import PaceDCMotorCfg
from pineapple_rl_lab.assets.assets.robots.cslrobotics import PINEAPPLE_V2_5_ARM_CFG  # isort: skip


LEG_JOINT_NAMES = ["L_hip_joint", "L_thigh_joint", "L_calf_joint", "R_hip_joint", "R_thigh_joint", "R_calf_joint"]
WHEEL_JOINT_NAMES = ["L_wheel_joint", "R_wheel_joint"]
ARM_JOINT_NAMES = [
    "arm_base_joint",
    "arm_link1_joint",
    "arm_link2_joint",
    "arm_link4_joint",
    "arm_link5_joint",
    "arm_link6_joint",
]
GRIPPER_JOINT_NAMES = ["arm_gripper_left_joint", "arm_gripper_right_joint"]

BASE_LINK_NAME = "base_link"
FOOT_LINK_NAME = ".*_wheel"
END_EFFECTOR_LINK_NAME = "arm_link6"

LEG_ACTION_DIM = len(LEG_JOINT_NAMES) + len(WHEEL_JOINT_NAMES)


def leg_action_rate_l2(env) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action[:, :LEG_ACTION_DIM] - env.action_manager.prev_action[:, :LEG_ACTION_DIM]),
        dim=1,
    )


def arm_action_rate_l2(env) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action[:, LEG_ACTION_DIM:] - env.action_manager.prev_action[:, LEG_ACTION_DIM:]),
        dim=1,
    )


def arm_action_smoothness_penalty(env) -> torch.Tensor:
    return torch.linalg.norm(env.action_manager.action[:, LEG_ACTION_DIM:] - env.action_manager.prev_action[:, LEG_ACTION_DIM:], dim=1)


def leg_action_smoothness_penalty(env) -> torch.Tensor:
    return torch.linalg.norm(env.action_manager.action[:, :LEG_ACTION_DIM] - env.action_manager.prev_action[:, :LEG_ACTION_DIM], dim=1)


@configclass
class PineappleArmActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        clip={".*_wheel_joint": (-25.13, 25.13)},
        preserve_order=True,
    )
    arm_pose = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class PineappleArmCommandsCfg:
    base_velocity = go2arm_mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        is_Go2ARM=True,
        curriculum_coeff=4000,
        ranges=go2arm_mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
        ),
        ranges_init=go2arm_mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        ranges_final=go2arm_mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
        ),
    )

    ee_pose = go2arm_mdp.command_cfg.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=END_EFFECTOR_LINK_NAME,
        resampling_time_range=(6.0, 8.0),
        debug_vis=True,
        is_Go2ARM=True,
        curriculum_start_iter=0,
        curriculum_coeff=5000,
        ranges=go2arm_mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.55),
            pos_y=(-0.2, 0.2),
            pos_z=(0.4, 0.65),
            roll=(0.0, 0.0),
            pitch=(0, math.pi / 9.0),
            yaw=(-math.pi / 12.0, math.pi / 12.0),
        ),
        ranges_init=go2arm_mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.45),
            pos_y=(0.0, 0.0),
            pos_z=(0.55, 0.55),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        ranges_final=go2arm_mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.55),
            pos_y=(-0.2, 0.2),
            pos_z=(0.4, 0.65),
            roll=(0.0, 0.0),
            pitch=(0, math.pi / 9.0),
            yaw=(-math.pi / 12.0, math.pi / 12.0),
        ),
    )


@configclass
class PineappleArmObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            history_length=10,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            history_length=10,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, history_length=10, params={"command_name": "base_velocity"})
        ee_pose_command = ObsTerm(func=mdp.generated_commands, history_length=10, params={"command_name": "ee_pose"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            scale=1.0,
            history_length=10,
            params={"asset_cfg": SceneEntityCfg(
                        "robot", 
                        joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES,)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            history_length=10,
            params={"asset_cfg": SceneEntityCfg(
                        "robot", 
                        joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES + ARM_JOINT_NAMES,)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action, history_length=10)

        priv_base_lin_vel = ObsTerm(func=go2arm_mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        priv_joint_torques = ObsTerm(
            func=go2arm_mdp.get_joints_torques,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES + ARM_JOINT_NAMES)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PineappleArmEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 1.5),
            "operation": "add",
        },
    )

    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=END_EFFECTOR_LINK_NAME),
            "mass_distribution_params": (-0.05, 0.2),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )


@configclass
class PineappleArmRewardsCfg:
    end_effector_position_tracking = RewardTermCfg(
        func=go2arm_mdp.position_command_error_exp,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=END_EFFECTOR_LINK_NAME), 
            "command_name": "ee_pose", 
            "std": 0.2,
        },
    )
    end_effector_orientation_tracking = RewardTermCfg(
        func=go2arm_mdp.orientation_command_error,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=END_EFFECTOR_LINK_NAME),
            "command_name": "ee_pose",
        },
    )
    end_effector_action_rate = RewardTermCfg(func=arm_action_rate_l2, weight=-0.02)
    end_effector_action_smoothness = RewardTermCfg(func=arm_action_smoothness_penalty, weight=-0.02)
    # end_effector_joint_acc = RewardTermCfg(
    #     func=mdp.joint_acc_l2,
    #     weight=-1e-6,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    # )
    # end_effector_joint_vel = RewardTermCfg(
    #     func=mdp.joint_vel_l2,
    #     weight=-1.0e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    # )
    # end_effector_joint_torques = RewardTermCfg(
    #     func=mdp.joint_torques_l2,
    #     weight=-1.0e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    # )

    is_terminated = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    base_linear_velocity = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    base_angular_velocity = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    action_rate = RewardTermCfg(func=leg_action_rate_l2, weight=-0.01)
    # action_smoothness = RewardTermCfg(func=leg_action_smoothness_penalty, weight=-0.01)
    # action_smoothness_2 = RewardTermCfg(
    #     func=pineapple_mdp.action_smoothness,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    # )
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-3.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_height = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.3},
    )
    joint_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc_wheel = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    joint_pos = RewardTermCfg(
        func=pineapple_mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_pos_hip = RewardTermCfg(
        func=pineapple_mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    joint_torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES)},
    )
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES)},
    )
    joint_alignment_hip = RewardTermCfg(
        func=pineapple_mdp.joint_align,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_a": "L_hip_joint", "joint_b": "R_hip_joint", "use": "pos", "opposite_sign": True},
    )
    joint_alignment_thigh = RewardTermCfg(
        func=pineapple_mdp.joint_align,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_a": "L_thigh_joint", "joint_b": "R_thigh_joint", "use": "pos"},
    )
    joint_alignment_calf = RewardTermCfg(
        func=pineapple_mdp.joint_align,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_a": "L_calf_joint", "joint_b": "R_calf_joint", "use": "pos"},
    )
    joint_pos_limit = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]), "threshold": 1.0},
    )
    joint_vel_limit = RewardTermCfg(
        func=mdp.joint_vel_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES), "soft_ratio": 0.9},
    )
    zero_cmd_lin_vel = RewardTermCfg(
        func=pineapple_mdp.zero_cmd_lin_vel,
        weight=-0.1,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    zero_cmd_ang_vel = RewardTermCfg(
        func=pineapple_mdp.zero_cmd_ang_vel,
        weight=-0.1,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class PineappleArmTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]), "threshold": 1.0},
    )


@configclass
class PineappleArmCurriculumCfg:
    pace_motor_delay = CurriculumTermCfg(
        func=pineapple_mdp.pace_motor_delay_curriculum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "actuator_names": ["hip", "thigh", "calf", "wheel"],
            "max_delay": 3,
            "distance_thresholds": (5.0, 7.5, 10.0),
        },
    )
    terrain_levels = CurriculumTermCfg(func=pineapple_mdp.terrain_levels_vel)


@configclass
class PineappleFlatEnvCfgV25(LocomotionVelocityRoughEnvCfg):
    observations: PineappleArmObservationsCfg = PineappleArmObservationsCfg()
    actions: PineappleArmActionsCfg = PineappleArmActionsCfg()
    commands: PineappleArmCommandsCfg = PineappleArmCommandsCfg()
    rewards: PineappleArmRewardsCfg = PineappleArmRewardsCfg()
    terminations: PineappleArmTerminationsCfg = PineappleArmTerminationsCfg()
    events: PineappleArmEventCfg = PineappleArmEventCfg()
    curriculum: PineappleArmCurriculumCfg = PineappleArmCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt

        self.scene.robot = PINEAPPLE_V2_5_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        self.curriculum.pace_motor_delay = None


class PineappleFlatEnvCfgV25_PLAY(PineappleFlatEnvCfgV25):
    def __post_init__(self) -> None:
        super().__post_init__()

        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.curriculum.pace_motor_delay = None

        self.commands.ee_pose.is_Go2ARM = False
        self.commands.ee_pose.is_Go2ARM_Play = True
        self.commands.ee_pose.resampling_time_range = (4.0, 4.0)
        # self.commands.ee_pose.ranges.pos_x = (0.25, 0.4)
        # self.commands.ee_pose.ranges.pos_y = (-0.35, 0.35)
        # self.commands.ee_pose.ranges.pos_z = (0.3, 0.75)
        self.commands.base_velocity.is_Go2ARM = False
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
