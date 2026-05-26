# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args_go2arm_style as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from local_rsl_rl_go2arm_style.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import export_policy_as_jit  # onnx not avaliable

#TODO:
from local_rsl_rl_go2arm_style.wrappers import RslRlVecEnvWrapper
from pineapple_rl_lab.tasks.manager_based.pineapple_rl_lab.agents_go2arm_style.rsl_rl_ppo_cfg_v2_5 import (
    Go2ArmRslRlOnPolicyRunnerCfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import numpy as np

import pineapple_rl_lab.tasks  # noqa: F401


def prepare_obs(env):
    """
        Modify the order of observations containing historical information for easier network reading. Try to avoid modifications if possible!

        Args:
            env: The environment
            agent_cfg: Agent configuration

        Returns:
            total(num_history, num_prop): Indices of the modified observation order
            obs_new(num_envs, num_prop * num_history): Modified observation order indices corresponding to the observations
        Notes:
            Try to avoid modifications if possible!

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_history) + joint_pos(18) * 10(num_history) + joint_vel(18) * 10(num_history)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
    """
    total = np.zeros((env.num_history, env.num_prop)) 
    obs_new = torch.zeros(env.num_envs, env.num_prop * env.num_history).to(env.device)
    
    lst, length = env.get_obs_list_length()
    lst = [item for item in lst if not item.startswith("policy-priv_")]

    result_dict = {}
    for i in range(len(lst)):
        c = np.array(list(range( sum(length[: i + 1 ]) - int(length[i] / env.num_history), sum(length[: i + 1]) )))
        result_dict[lst[i]] = c

    key_list = list(result_dict.keys())
    a1_list = []
    for i in range(env.num_history):
        for j in range(len(lst)):
            a1 = np.concatenate([
                result_dict[key_list[j]] - (i) * result_dict[key_list[j]].shape[0]])
            a1_list.append(a1)
            if j == len(lst) - 1:
                a1_list = np.concatenate(a1_list)
                total[i, :] = a1_list
                a1_list = []
    return total, obs_new


def _ids_to_list(ids):
    if isinstance(ids, slice):
        return ids
    if torch.is_tensor(ids):
        return ids.detach().cpu().tolist()
    return list(ids)


def print_debug_index_map(env):
    robot = env.unwrapped.scene["robot"]
    robot_joint_names = getattr(robot, "joint_names", None)
    if robot_joint_names is None:
        robot_joint_names = getattr(robot.data, "joint_names", None)

    print("\n=== Play debug: policy obs index map ===")
    cursor = 0
    for term_name, term_value in env.unwrapped.observation_manager.get_active_iterable_terms(0):
        if term_name.startswith("policy-priv_"):
            continue
        width = len(term_value)
        per_step_width = int(width / env.num_history)
        print(f"{term_name:28s} raw=[{cursor}:{cursor + width}) per_step={per_step_width}")
        cursor += width

    print("\n=== Play debug: resolved obs joints ===")
    obs_term_names = env.unwrapped.observation_manager._group_obs_term_names.get("policy", [])
    obs_term_cfgs = env.unwrapped.observation_manager._group_obs_term_cfgs.get("policy", [])
    for obs_name, obs_cfg in zip(obs_term_names, obs_term_cfgs):
        asset_cfg = getattr(obs_cfg, "params", {}).get("asset_cfg") if hasattr(obs_cfg, "params") else None
        if asset_cfg is None or not hasattr(asset_cfg, "joint_ids") or robot_joint_names is None:
            continue

        joint_ids = _ids_to_list(asset_cfg.joint_ids)
        if isinstance(joint_ids, slice):
            resolved_names = robot_joint_names[joint_ids]
        else:
            resolved_names = [robot_joint_names[joint_id] for joint_id in joint_ids]
        print(f"{obs_name:20s} joint_ids={joint_ids} joints={resolved_names}")

    print("\n=== Play debug: action index map ===")
    cursor = 0
    action_terms = env.unwrapped.action_manager._terms
    action_items = action_terms.items() if hasattr(action_terms, "items") else enumerate(action_terms)
    for term_name, term in action_items:
        joint_ids = _ids_to_list(getattr(term, "_joint_ids", []))
        if robot_joint_names is not None:
            joint_names = [robot_joint_names[joint_id] for joint_id in joint_ids]
        else:
            joint_names = joint_ids
        width = len(joint_ids)
        print(f"{term_name:16s} action=[{cursor}:{cursor + width}) joints={joint_names}")
        cursor += width


def print_debug_arm_state(env, actions, step):
    robot = env.unwrapped.scene["robot"]
    robot_joint_names = getattr(robot, "joint_names", None)
    if robot_joint_names is None:
        robot_joint_names = getattr(robot.data, "joint_names", None)
    if robot_joint_names is None:
        print("[WARN] Play debug: robot joint names unavailable.")
        return

    arm_names = [
        "arm_base_joint",
        "arm_link1_joint",
        "arm_link2_joint",
        "arm_link4_joint",
        "arm_link5_joint",
        "arm_link6_joint",
    ]
    arm_ids = [robot_joint_names.index(name) for name in arm_names]
    arm_action_start = 8
    arm_actions = actions[0, arm_action_start : arm_action_start + len(arm_names)].detach().cpu()
    arm_pos = robot.data.joint_pos[0, arm_ids].detach().cpu()
    arm_default = robot.data.default_joint_pos[0, arm_ids].detach().cpu()
    arm_vel = robot.data.joint_vel[0, arm_ids].detach().cpu()

    print(f"\n=== Play debug: arm state step={step} env0 ===")
    for idx, name in enumerate(arm_names):
        print(
            f"action[{arm_action_start + idx:02d}] {name:18s} "
            f"act={arm_actions[idx]: .4f} pos={arm_pos[idx]: .4f} "
            f"default={arm_default[idx]: .4f} delta={arm_pos[idx] - arm_default[idx]: .4f} "
            f"vel={arm_vel[idx]: .4f}"
        )


def change_obs_order(obs, obs_new, total, env):

    """
        Modify the order of observations containing historical information for easier network reading. Try to avoid modifications if possible!

        Args:
            obs(num_envs, num_prop * num_history): The input to the actor and critic network
            obs_new(num_envs, num_prop * num_history): Modified observation order indices corresponding to the observations
            total(num_history, num_prop): Indices of the modified observation order
            env: The environment
            agent_cfg: Agent configuration

        Returns:
            obs(num_envs, num_prop * num_history): The input to the actor and critic network
            obs_new(num_envs, num_prop * num_history): Modified observation order indices corresponding to the observations(need to reset)

        Notes:
            Try to avoid modifications if possible!

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_history) + joint_pos(18) * 10(num_history) + joint_vel(18) * 10(num_history)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
    """
    for i in range(env.num_history):
        obs_1 = obs[:, total[i, :]].to(env.device)
        obs_new = torch.cat([obs_new, obs_1], dim = -1)

    # only prop obs:    
    obs = obs_new[:, env.num_prop  * env.num_history:] 
    
    # prop and priv obs:
    # obs = torch.cat([obs_new[:, env.num_prop  * env.num_history :], 
    #                  obs[:, env.num_prop  * env.num_history :]], dim=-1)  
    
    obs_new = torch.zeros(env.num_envs, env.num_prop * env.num_history).to(env.device)
    return obs, obs_new



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Go2ArmRslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    
    # onnx not avaliable
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    total, obs_new = prepare_obs(env)
    print_debug_index_map(env)
    
    timestep = 0
    debug_step = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs, obs_new = change_obs_order(obs, obs_new, total, env)

            actions = policy(obs,hist_encoding=True) #no priv obs
            if debug_step % 100 == 0:
                print_debug_arm_state(env, actions, debug_step)
            # env stepping
            obs, _, _, _,_ = env.step(actions)
            debug_step += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
