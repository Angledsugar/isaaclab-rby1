# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation for the RBY1-M mecanum-wheel mobile base."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleop for RBY1-M.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Template-Rby1-Nav-v0", help="Task name.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=500, help="Video length in steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable offscreen rendering for video
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rby1.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    # Wrap with video recorder if requested
    if args_cli.video:
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array",
        )
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="videos",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="rby1_teleop",
        )
        print(f"[INFO] Recording video to ./videos/ ({args_cli.video_length} steps)")
    else:
        env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Action space: {env.action_space}")
    print(f"[INFO]: Observation space: {env.observation_space}")

    # Setup keyboard: Se2Keyboard gives (v_x, v_y, omega_z)
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=4.0,
        v_y_sensitivity=4.0,
        omega_z_sensitivity=4.0,
        sim_device=env.unwrapped.device,
    )
    keyboard = Se2Keyboard(keyboard_cfg)
    print(keyboard)

    env.reset()
    keyboard.reset()

    print("\n=== RBY1-M Mecanum Teleop ===")
    print("  Arrow Up/Down    : forward/backward")
    print("  Arrow Left/Right : strafe left/right")
    print("  Z / X            : rotate left/right")
    print("  L                : stop all")
    print("=============================\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            # Se2Keyboard: (v_x, v_y, omega_z)
            se2_cmd = keyboard.advance()  # (3,)
            vx = se2_cmd[0].item()
            vy = se2_cmd[1].item()
            wz = se2_cmd[2].item()

            # Mecanum wheel inverse kinematics:
            #   wheel_fl = vx - vy - wz
            #   wheel_fr = vx + vy + wz
            #   wheel_rl = vx + vy - wz
            #   wheel_rr = vx - vy + wz
            fl = vx - vy - wz
            fr = vx + vy + wz
            rl = vx + vy - wz
            rr = vx - vy + wz

            actions = torch.tensor([[fl, fr, rl, rr]], device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
