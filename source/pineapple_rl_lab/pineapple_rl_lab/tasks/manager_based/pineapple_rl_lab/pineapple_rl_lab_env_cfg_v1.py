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
from pineapple_rl_lab.assets.assets.robots.cslrobotics import PINEAPPLE_V0_CFG  # isort: skip

LEG_JOINT_NAMES = ["L_thigh_joint", "L_calf_joint", "R_thigh_joint", "R_calf_joint"]
WHEEL_JOINT_NAMES = ["L_wheel_joint", "R_wheel_joint"]
BASE_LINK_NAME = "base_link"
FOOT_LINK_NAME = ".*_wheel_link"

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class PineappleActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.5, use_default_offset=True)
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEEL_JOINT_NAMES, scale=5.0, use_default_offset=True)


@configclass
class PineappleCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-3.14, 3.14)
        ),
    )


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
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

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

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
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
        func=mdp.reset_joints_by_scale,
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
    # is_terminated = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    action_smoothness = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_2 = RewardTermCfg(
        func=pineapple_mdp.action_smoothness, 
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_orientation = RewardTermCfg(
        func=mdp.flat_orientation_l2, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_height = RewardTermCfg(
        func=mdp.base_height_l2, 
        weight=-20.0, 
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.1871}
    )
    joint_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc_wheel = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    joint_pos = RewardTermCfg(
        func=pineapple_mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
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

    # joint_vel_limit = RewardTermCfg(
    #     func=mdp.joint_vel_limits,
    #     weight=-10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES), "soft_ratio":0.9},
    # )

    # joint_vel_limit_wheel = RewardTermCfg(
    #     func=mdp.joint_vel_limits,
    #     weight=-10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES), "soft_ratio":0.8},
    # )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 1.0,
        },
    )


@configclass
class PineappleTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]), "threshold": 1.0},
    )
    # terrain_out_of_bounds = DoneTerm(
    #     func=mdp.terrain_out_of_bounds,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    #     time_out=True,
    # )


@configclass
class PineappleCurriculumCfg:
    """Curriculum terms for the MDP."""

    velocity_command = CurriculumTermCfg(
        func=pineapple_mdp.velocity_command_curriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "base_linear_velocity",
            "x_vel_range": (-2.0, 2.0),
            "z_ang_vel_range": (-3.14, 3.14),
            "reward_threshold_ratio": 0.9,
            "range_step_size": 0.1,
        },
    )


@configclass
class PineappleFlatEnvCfgV1(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: PineappleObservationsCfg = PineappleObservationsCfg()
    actions: PineappleActionsCfg = PineappleActionsCfg()
    commands: PineappleCommandsCfg = PineappleCommandsCfg()

    # MDP setting
    rewards: PineappleRewardsCfg = PineappleRewardsCfg()
    terminations: PineappleTerminationsCfg = PineappleTerminationsCfg()
    events: PineappleEventCfg = PineappleEventCfg()
    # curriculum: PineappleCurriculumCfg = PineappleCurriculumCfg()

    # Viewer
    # viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Pineapple
        self.scene.robot = PINEAPPLE_V0_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
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


class PineappleFlatEnvCfgV1_PLAY(PineappleFlatEnvCfgV1):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

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
