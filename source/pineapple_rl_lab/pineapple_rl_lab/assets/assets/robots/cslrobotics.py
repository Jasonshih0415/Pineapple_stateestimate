import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from pineapple_rl_lab.assets.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


PINEAPPLE_V0_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/csl/pineapplev0_description/urdf/quick_bipedal.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "L_thigh_joint": 1.401,
            "L_calf_joint": -2.0717,
            "L_wheel_joint": 0.0,
            "R_thigh_joint": 1.401,
            "R_calf_joint": -2.0717,
            "R_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "thigh": DCMotorCfg(
            joint_names_expr=["L_thigh_joint", "R_thigh_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=["L_calf_joint", "R_calf_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=["L_wheel_joint", "R_wheel_joint"],
            effort_limit=3.69,
            saturation_effort=3.69,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.3,
            friction=0.0,
        ),
    },
)


PINEAPPLE_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/csl/pineapple/urdf/pineapple.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "L_hip_joint": 0.0,
            "L_thigh_joint": 1.2533,
            "L_calf_joint": -2.0479,
            "L_wheel_joint": 0.0,
            "R_hip_joint": 0.0,
            "R_thigh_joint": 1.2533,
            "R_calf_joint": -2.0479,
            "R_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=["L_hip_joint", "R_hip_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=50.0,
            damping=0.5,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=["L_thigh_joint", "R_thigh_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=["L_calf_joint", "R_calf_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=["L_wheel_joint", "R_wheel_joint"],
            effort_limit=3.69,
            saturation_effort=3.69,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.3,
            friction=0.0,
        ),
    },
)