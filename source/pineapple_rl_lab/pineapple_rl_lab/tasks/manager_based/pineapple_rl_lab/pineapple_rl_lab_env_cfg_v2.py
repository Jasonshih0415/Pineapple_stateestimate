import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import math

import pineapple_rl_lab.tasks.manager_based.pineapple_rl_lab.mdp as pineapple_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from pineapple_rl_lab.assets.assets.robots.cslrobotics import PINEAPPLE_V2_CFG  # isort: skip

LEG_JOINT_NAMES = ["L_hip_joint", "L_thigh_joint", "L_calf_joint", "R_hip_joint", "R_thigh_joint", "R_calf_joint"]
WHEEL_JOINT_NAMES = ["L_wheel_joint", "R_wheel_joint"]

BASE_LINK_NAME = "base_link"
FOOT_LINK_NAME = ".*_wheel"

@configclass
class PineappleActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.5, use_default_offset=True)
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEEL_JOINT_NAMES, scale=5.0, use_default_offset=True, clip={".*_wheel_joint": (-25.13, 25.13)})


@configclass 
class PineappleCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = pineapple_mdp.PineappleVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Calculated based on Wheel Velocity Limit (25.13 rad/s), Radius (0.077m), and Half-Track-Width (~0.205m)
            # Max Lin Vel = 25.13 * 0.077 = ~1.93 m/s
            # Max Ang Vel = 1.93 / 0.205 = ~9.4 rad/s
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.0, 2.0)
        ),
    )

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 15.0),
    #     rel_standing_envs=0.1,
    #     rel_heading_envs=0.0,
    #     heading_command=False,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.5, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.0, 2.0)
    #     ),
    # )


@configclass
class PineappleObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        # )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            scale=0.25,
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}, 
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6
            self.flatten_history_dim = True
            
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, scale=2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.25, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, scale=1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PineappleEventCfg:
    """Configuration for randomization."""

    # startup
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

    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.2, 0.5),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
    #         "force_range": (-1.0, 1.0),
    #         "torque_range": (-0.5, 0.5),
    #     },
    # )

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


    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )


@configclass
class PineappleRewardsCfg:
    # -- task
    base_linear_velocity = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    base_angular_velocity = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    # is_terminated = RewardTermCfg(func=mdp.is_terminated, weight=-500.0)
    action_smoothness = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_2 = RewardTermCfg(
        func=pineapple_mdp.action_smoothness, 
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_orientation = RewardTermCfg(
        func=mdp.flat_orientation_l2, weight=-20.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_height = RewardTermCfg(
        func=mdp.base_height_l2, 
        weight=-5.0, 
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.3}
    )
    joint_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc_wheel = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5e-6,
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
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_alignment_hip = RewardTermCfg(
        func=pineapple_mdp.joint_align,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_a": "L_hip_joint",
            "joint_b": "R_hip_joint",
            "use": "pos",
            "opposite_sign": True,
            },
    )
    joint_alignment_thigh = RewardTermCfg(
        func=pineapple_mdp.joint_align,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_a": "L_thigh_joint",
            "joint_b": "R_thigh_joint",
            "use": "pos"
            },
    )
    joint_alignment_calf = RewardTermCfg(
        func=pineapple_mdp.joint_align,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_a": "L_calf_joint",
            "joint_b": "R_calf_joint",
            "use": "pos"
            },
    )
    joint_pos_limit = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 1.0,
        },
    )
    joint_vel_limit = RewardTermCfg(
        func=mdp.joint_vel_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio":0.9},
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
class PineappleTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]), "threshold": 1.0},
    )


@configclass
class PineappleCurriculumCfg:
    """Curriculum terms for the MDP."""

    pace_motor_delay = CurriculumTermCfg(
        func=pineapple_mdp.pace_motor_delay_curriculum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "actuator_names": ["hip", "thigh", "calf", "wheel"],
            "max_delay": 3,
            # mean episode distance (m) at which delay increments: 0→1, 1→2
            "distance_thresholds": (5.0, 7.5, 10.0),
        },
    )
    terrain_levels = CurriculumTermCfg(func=pineapple_mdp.terrain_levels_vel)

@configclass
class PineappleFlatEnvCfgV2(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: PineappleObservationsCfg = PineappleObservationsCfg()
    actions: PineappleActionsCfg = PineappleActionsCfg()
    commands: PineappleCommandsCfg = PineappleCommandsCfg()

    # MDP setting
    rewards: PineappleRewardsCfg = PineappleRewardsCfg()
    terminations: PineappleTerminationsCfg = PineappleTerminationsCfg()
    events: PineappleEventCfg = PineappleEventCfg()
    curriculum: PineappleCurriculumCfg = PineappleCurriculumCfg()

    # Viewer
    # viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Pineapple
        self.scene.robot = PINEAPPLE_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # no height scan
        self.scene.height_scanner = None


class PineappleFlatEnvCfgV2_PLAY(PineappleFlatEnvCfgV2):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # set velocity command ranges for play
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event

        # disable delay curriculum for play
        self.curriculum.pace_motor_delay = None
