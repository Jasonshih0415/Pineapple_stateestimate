import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
import gui_teleop
from mujoco_sipo_v3 import SIPO, get_contact_states 

def calculate_com_in_base_frame(model, data, base_body_id):
    total_mass = 0.0
    com_sum = np.zeros(3)

    # Get base position and rotation
    base_pos = data.xipos[base_body_id]  # Position of the base in world coordinates
    base_rot = data.ximat[base_body_id].reshape(3, 3)  # Rotation matrix of the base

    for i in range(model.nbody):
        # Get body mass and world COM position
        mass = model.body_mass[i]
        world_com = data.xipos[i]

        # Transform COM to base coordinates
        local_com = world_com - base_pos  # Translate to base origin
        local_com = base_rot.T @ local_com  # Rotate into base frame

        # Accumulate mass-weighted positions
        com_sum += mass * local_com
        total_mass += mass

    # Compute overall COM in base coordinates
    center_of_mass_base = com_sum / total_mass
    return center_of_mass_base

def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    
    Args:
        q: The quaternion in (w, x, y, z) format. Shape is (..., 4).
        v: The vector in (x, y, z) format. Shape is (..., 3).
        
    Returns:
        The rotated vector in (x, y, z) format. Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Equivalent to (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded
    
    # Equivalent to torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0
    
    # Equivalent to the torch.bmm or torch.einsum operations
    # This calculates the dot product between q_vec and v
    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0
    
    return a - b + c

def get_gravity_orientation(quaternion):
    """
    Get the gravity vector in the robot's base frame.
    Uses the exact algorithm from your PyTorch code.
    
    Args:
        quaternion: Quaternion in (w, x, y, z) format.
        
    Returns:
        3D gravity vector in the robot's base frame.
    """
    # # Ensure quaternion is a numpy array
    # quaternion = np.array(quaternion)
    
    # # Standard gravity vector in world frame (pointing down)
    # gravity_world = np.array([0, 0, -1])
    
    # # Handle both single quaternion and batched quaternions
    # if quaternion.shape == (4,):
    #     quaternion = quaternion.reshape(1, 4)
    #     gravity_world = gravity_world.reshape(1, 3)
    #     result = quat_rotate_inverse(quaternion, gravity_world)[0]
    # else:
    #     gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
    #     result = quat_rotate_inverse(quaternion, gravity_world)
    q = np.array(quaternion)
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0]=2*(-q[1]*q[3]+q[0]*q[2])
    gravity[1]=-2*(q[2]*q[3]+q[0]*q[1])
    gravity[2]=1-2*(q[0]*q[0]+q[3]*q[3])
    return gravity
    # return result

def apply_diamond_constraint(cmd, max_lin, max_ang):
    """
    Applies a diamond-shaped constraint (L1 norm) on the linear and angular velocities:
    |v_x| / v_max + |w_z| / w_max <= 1
    """
    vx = cmd[0]
    wz = cmd[2]
    
    limit_vx = max_lin
    limit_wz = max_ang
    
    if limit_vx < 1e-6: limit_vx = 1.0
    if limit_wz < 1e-6: limit_wz = 1.0
    
    ratio = abs(vx) / limit_vx + abs(wz) / limit_wz
    
    if ratio > 1.0:
        scaling = 1.0 / ratio
        cmd[0] *= scaling
        cmd[2] *= scaling
    return cmd

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        policy = torch.jit.load(policy_path)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        pos_action_scale = config["pos_action_scale"]
        vel_action_scale = config["vel_action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config.get("obs_buffer_size", 1)

        leg_joint_indices = config["leg_joint_indices"]
        wheel_joint_indices = config["wheel_joint_indices"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        max_lin = config.get("max_lin_vel", 1.0)
        max_ang = config.get("max_ang_vel", 1.0)
        
        # Height Command
        height_scale = config.get("height_scale", 1.0)
        cmd_height_init = config.get("cmd_height_init", 0.3)
        min_height = config.get("min_height", 0.2)
        max_height = config.get("max_height", 0.35)
        height_step = config.get("height_step", 0.005)

        policy_index_map = config.get("policy_index_map", None)
        if policy_index_map is not None:
             policy_index_map = np.array(policy_index_map, dtype=np.int64)

    # Initialize Gamepad
    # cmd_vel = np.array(config["cmd_init"], dtype=np.float32)
    teleop = gui_teleop.GUITeleop(
        config_init=config["cmd_init"], 
        max_lin=max_lin, 
        max_ang=max_ang,
        height_init=cmd_height_init,
        height_step=height_step,
        min_height=min_height,
        max_height=max_height
    )
    # gamepad = gamepad_reader.Gamepad(vel_scale_x=1.0, vel_scale_y=1.0, vel_scale_rot=3.0)
    print("GUI Teleop initialized. Click the 'Control Panel' window to send commands.")

    target_dof_pos = default_angles.copy()
    target_dof_vel = np.zeros(num_actions)
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history_buffer = torch.zeros((obs_buffer_size, one_step_obs_size))

    # gamepad = gamepad_reader.Gamepad(vel_scale_x=0.5, vel_scale_y=0.5, vel_scale_rot=0.8)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1

    # Record data
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    action_list = []
    time_list = []
    cmd_list = []
    
    # SIPO Record
    sipo_pos_list = []
    sipo_vel_list = []
    sipo_quat_list = [] # SIPO estimated quaternion (World frame)
    sipo_vel_body_list = []
    gt_pos_list = []
    gt_vel_list = []
    gt_quat_list = [] # GT quaternion (World frame)
    gt_vel_body_list = []
    
    imu_yaw_rate_list = []
    sipo_yaw_rate_list = []
    gt_yaw_rate_list = []
    
    
    # time_list = [i * simulation_dt for i in range(int(simulation_duration / simulation_dt))]

    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        mujoco.mj_resetDataKeyframe(m, d, 0)
        viewer.sync()
        
        # --- Initialize SIPO (After Reset) ---
        sipo = SIPO(xml_path)
        # Note: Data has been reset to keyframe 0.
        # We need to make sure d.xipos etc are updated before we read them.
        mujoco.mj_forward(m, d) # Update kinematics/position data from qpos
        
        gt_pos = d.xipos[base_body_id].copy()
        gt_quat = d.xquat[base_body_id].copy()
        
        # Initial Feet
        # Need qpos/qvel from sensordata. But sensordata might not be computed yet?
        # mj_forward computes sensor data if sensors are enabled.
        # Check if d.sensordata is populated.
        # If not, use d.qpos directly to fake it.
        # Map d.qpos (15 dim) to sensor qpos (8 dim).
        # qpos structure: Free(7), L_hip, L_thigh, L_calf, L_wheel, R_hip...
        # Indices: 7,8,9,10, 11,12,13,14. (Total 8).
        # Assuming order L then R.
        
        init_qpos_sense = d.qpos[7:15].copy()
        init_qvel_sense = d.qvel[6:14].copy()
        
        z_kin = sipo.get_kinematics(init_qpos_sense, init_qvel_sense)
        # z_kin contains [p1, v1, p2, v2...] (6 per leg). Extract positions.
        feet_pos_flat = z_kin.reshape(sipo.num_legs, 6)[:, :3].flatten()
        
        sipo.init_state(gt_pos, gt_quat, feet_pos_flat)
        print("SIPO Initialized after Reset.")
        # -------------------------------------

        while viewer.is_running() and counter*simulation_dt < simulation_duration:
            step_start = time.time()
    
            tau = pd_control(target_dof_pos, d.sensordata[:num_actions], kps, target_dof_vel, d.sensordata[num_actions:num_actions + num_actions], kds)
            
            # if time.time() - start > 3:
            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            # com_base = calculate_com_in_base_frame(m, d, base_body_id)
            # print("Center of Mass in Base Coordinates:", com_base)

            # After mujoco.mj_step(m, d)
            robot_pos = d.xipos[base_body_id]  # Get robot base position in world coordinates
            viewer.cam.lookat[:] = robot_pos   # Set camera to look at robot base

            counter += 1
            # qvel = d.sensordata[6:12]
            # joint_vel_list.append(qvel.copy())
            # action_list.append(target_dof_vel.copy())
            
            # create observation
            qpos = d.sensordata[:num_actions]
            qvel = d.sensordata[num_actions: 2*num_actions]
            imu_quat = d.sensordata[3*num_actions:3*num_actions+4]
            ang_vel_b = d.sensordata[3*num_actions+4:3*num_actions+4+3]
            imu_acc = d.sensordata[3*num_actions+4+3:3*num_actions+4+3+3] # Added extraction

            lin_vel_I = d.sensordata[3*num_actions+4+3+6:3*num_actions+4+3+6+3]

            # --- SIPO Update ---
            contacts = get_contact_states(m, d, sipo.leg_names)
            contacts.fill(1.0) # Force Full Contact
            
            # 【修正】: 將 qpos (即腿部與輪子的關節角度) 傳入 predict，用來算 R_hub
            sipo.predict(imu_acc, ang_vel_b, simulation_dt, qpos)
            # Yaw measurement from GT quat for now (or None if drifting is okay)
            # Using ground truth yaw to stabilize heading like mocap
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, d.xquat[base_body_id])
            yaw_gt = np.arctan2(mat[3], mat[0]) # Simplified
            # Better yaw extraction:
            gt_q_w = d.xquat[base_body_id][0]
            gt_q_z = d.xquat[base_body_id][3]
            # Simple yaw from quat (assuming small pitch/roll)
            # yaw ≈ atan2(2(wz + xy), 1 - 2(y^2 + z^2)) 
            # Or just pass None to test drift.
            
            wheel_vel_meas = qvel[wheel_joint_indices]
            sipo_state = sipo.update(qpos, qvel, contacts, ang_vel_b, wheel_vel_meas, yaw_meas=None)
            
            sipo_pos_list.append(sipo_state[sipo.idx_pos].copy())
            sipo_vel_list.append(sipo_state[sipo.idx_vel].copy())
            sipo_vel_body_list.append(quat_rotate_inverse(sipo_state[sipo.idx_quat], sipo_state[sipo.idx_vel]))
            sipo_quat_list.append(sipo_state[sipo.idx_quat].copy())
            gt_pos_list.append(d.xipos[base_body_id].copy())
            gt_vel_list.append(d.cvel[base_body_id][3:6].copy()) # cvel is rotational(0:3), translational(3:6)
            gt_vel_body_list.append(quat_rotate_inverse(d.xquat[base_body_id], d.cvel[base_body_id][3:6]))
            gt_quat_list.append(d.xquat[base_body_id].copy())
            
            # Record Yaw Rate (Z-axis)
            # IMU Raw
            imu_yaw_rate_list.append(ang_vel_b[2])
            
            # SIPO Corrected (Raw - Bias[Z])
            # Check sipo.idx_bg relative to state vector
            bg = sipo_state[sipo.idx_bg]
            sipo_yaw_rate_list.append(ang_vel_b[2] - bg[2])
            
            # Ground Truth (Body Frame)
            # Need to transform World (cvel 0:3) to Body.
            gt_w_world = d.cvel[base_body_id][0:3].copy()
            # quat is [w, x, y, z]
            gt_w_body = quat_rotate_inverse(d.xquat[base_body_id], gt_w_world)
            gt_yaw_rate_list.append(gt_w_body[2])
            # -------------------

            if policy_index_map is not None:
                qpos_obs = qpos[policy_index_map]
                qvel_obs = qvel[policy_index_map]
                default_angles_pol = default_angles[policy_index_map]
            else:
                qpos_obs = qpos
                qvel_obs = qvel
                default_angles_pol = default_angles

            current_cmd_vel = np.array(teleop.get_command(), dtype=np.float32)
            current_cmd_vel = apply_diamond_constraint(current_cmd_vel, max_lin, max_ang)
            
            # Retrieve height command
            current_cmd_height = np.array([teleop.get_height_command()], dtype=np.float32)

            lin_vel_b = quat_rotate_inverse(imu_quat, lin_vel_I)
            gravity_b = get_gravity_orientation(imu_quat)

            # SAFE leg joint delta (1D)
            valid_leg_idx = [i for i in leg_joint_indices if i < len(qpos_obs) and i < len(default_angles_pol)]
            leg_pos_delta = (qpos_obs[valid_leg_idx] - default_angles_pol[valid_leg_idx]) * dof_pos_scale
            leg_pos_delta = leg_pos_delta.astype(np.float32).ravel()
            
            # NOTE: Added current_cmd_height after current_cmd_vel
            obs_list = [
                ang_vel_b * ang_vel_scale,
                gravity_b,
                current_cmd_vel * cmd_scale,
                current_cmd_height * height_scale,
                leg_pos_delta,
                qvel_obs * dof_vel_scale,
                action.astype(np.float32)
            ]
            ## Record Data ##
            lin_vel_data_list.append(lin_vel_b.copy())
            ang_vel_data_list.append(ang_vel_b.copy())
            gravity_b_list.append(gravity_b)
            joint_vel_list.append(qvel_obs.copy()) 
            action_list.append(action * vel_action_scale)
            time_list.append(counter * simulation_dt)
            cmd_list.append(current_cmd_vel.copy()) 
            ###

            if counter % control_decimation == 0 and counter > 0:

                obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]

                current_obs = torch.cat(obs_list, dim=0)

                obs_history_buffer = torch.roll(obs_history_buffer, shifts=-1, dims=0)
                obs_history_buffer[-1] = current_obs

                # Stack by feature group: [feat1_t, feat1_t-1, ..., feat2_t, ...]
                split_sizes = [o.numel() for o in obs_list]
                feature_groups = torch.split(obs_history_buffer, split_sizes, dim=1)
                flat_groups = [g.flatten() for g in feature_groups]
                
                obs_tensor_buf = torch.cat(flat_groups).unsqueeze(0)
                obs_tensor_buf = torch.clip(obs_tensor_buf, -100, 100)

                # obs inference
                action = policy(obs_tensor_buf).detach().numpy().squeeze()

                # Set leg joint target positions
                for idx in leg_joint_indices:       # 0 1 2 3 4 5
                    if policy_index_map is not None:
                        idx_xml = policy_index_map[idx]
                    else:
                        idx_xml = idx

                    if idx_xml < len(target_dof_pos) and idx < len(action):
                        target_dof_pos[idx_xml] = default_angles[idx_xml] + action[idx] * pos_action_scale
                # Set wheel joint target velocities
                for idx in wheel_joint_indices:
                    if policy_index_map is not None:
                        idx_xml = policy_index_map[idx]
                    else:
                        idx_xml = idx

                    if idx_xml < len(target_dof_vel) and idx < len(action):
                        target_dof_vel[idx_xml] = action[idx] * vel_action_scale

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Plot the collected data after the simulation ends
    # --- SIPO Results Figure ---
    fig_sipo = plt.figure(figsize=(12, 20)) # Increase height
    t = time_list
    
    # Calculate Body Velocities for plotting (Removed: Now done in real-time)

    # Top: 2D XY Plane (Trajectory) - Position [1]
    ax_traj = plt.subplot(5, 1, 1) # Change to 5 rows
    ax_traj.set_title("2D Position Trajectory (XY Plane)")
    ax_traj.plot([p[0] for p in sipo_pos_list], [p[1] for p in sipo_pos_list], label='SIPO')
    ax_traj.plot([p[0] for p in gt_pos_list], [p[1] for p in gt_pos_list], label='GT', linestyle='--')
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend()
    ax_traj.grid()
    ax_traj.axis('equal')
    
    # Second Row: Velocity X & Y (World Frame)
    ax_vx = plt.subplot(5, 2, 3) 
    ax_vx.set_title("Velocity X (World)")
    ax_vx.plot(t, [v[0] for v in sipo_vel_list], label='SIPO')
    ax_vx.plot(t, [v[0] for v in gt_vel_list], label='GT', linestyle='--')
    ax_vx.set_xlabel("Time (s)")
    ax_vx.legend()
    ax_vx.grid()

    ax_vy = plt.subplot(5, 2, 4)
    ax_vy.set_title("Velocity Y (World)")
    ax_vy.plot(t, [v[1] for v in sipo_vel_list], label='SIPO')
    ax_vy.plot(t, [v[1] for v in gt_vel_list], label='GT', linestyle='--')
    ax_vy.set_xlabel("Time (s)")
    ax_vy.legend()
    ax_vy.grid()

    # Third Row: Velocity X & Y (Body Frame) <--- NEW
    ax_vbx = plt.subplot(5, 2, 5) 
    ax_vbx.set_title("Velocity X (Body/Robot Frame)")
    ax_vbx.plot(t, [v[0] for v in sipo_vel_body_list], label='SIPO Body')
    ax_vbx.plot(t, [v[0] for v in gt_vel_body_list], label='GT Body', linestyle='--')
    ax_vbx.set_xlabel("Time (s)")
    ax_vbx.legend()
    ax_vbx.grid()

    ax_vby = plt.subplot(5, 2, 6)
    ax_vby.set_title("Velocity Y (Body/Robot Frame)")
    ax_vby.plot(t, [v[1] for v in sipo_vel_body_list], label='SIPO Body')
    ax_vby.plot(t, [v[1] for v in gt_vel_body_list], label='GT Body', linestyle='--')
    ax_vby.set_xlabel("Time (s)")
    ax_vby.legend()
    ax_vby.grid()
    
    # Fourth Row: Yaw Angular Velocity
    ax_yaw = plt.subplot(5, 1, 4)
    ax_yaw.set_title("Yaw Angular Velocity (Z-axis) [Rad/s]")
    ax_yaw.plot(t, imu_yaw_rate_list, label='IMU Raw', color='purple', alpha=0.5, linewidth=1.0)
    ax_yaw.plot(t, sipo_yaw_rate_list, label='SIPO Est (Corrected)', color='blue', linewidth=1.5)
    ax_yaw.plot(t, gt_yaw_rate_list, label='GT Body', color='green', linestyle='--', linewidth=1.5)
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.legend()
    ax_yaw.grid()

    # Fifth Row: Z Height - Position [4]
    ax_z = plt.subplot(5, 1, 5)
    ax_z.set_title("Z Height (Position Z) [m]")
    # Z is index 2 in position vector (x, y, z)
    ax_z.plot(t, [p[2] + sipo.wheel_radius for p in sipo_pos_list], label='SIPO')
    ax_z.plot(t, [p[2] for p in gt_pos_list], label='GT', linestyle='--')
    ax_z.set_xlabel("Time (s)")
    ax_z.legend()
    ax_z.grid()
    
    plt.tight_layout()
    plt.savefig("sipo_results.png", dpi=300)
    plt.close(fig_sipo)

    # --- History Data Figure ---
    fig_hist = plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    for i in range(3): 
        plt.plot(time_list, [step[i] for step in lin_vel_data_list], label=f"Linear Velocity {i}")
    plt.plot(time_list, [step[0] for step in cmd_list], label=f"Command Velocity x", linestyle='--')
    plt.title(f"History Linear Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.plot(time_list, [step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.plot(time_list, [step[2] for step in cmd_list], label=f"Command Velocity yaw", linestyle='--')
    plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    for i in range(3):
        plt.plot(time_list, [step[i] for step in gravity_b_list], label=f"Project Gravity {i}")
    plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    for i in wheel_joint_indices:
        plt.plot(time_list, [step[i] for step in joint_vel_list], label=f"Joint Velocity {i}")
    for i in wheel_joint_indices:
        plt.plot(time_list, [step[i] for step in action_list], label=f"velocity Command {i}", linestyle='--')
    plt.title(f"History Joint Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("history_data.png", dpi=300)
    plt.close(fig_hist)
    # # plt.show()

    teleop.close()