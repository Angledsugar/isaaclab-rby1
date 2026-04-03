# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from rby1.assets import RBY1_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class Rby1NavEnvCfg(DirectRLEnvCfg):
    """Configuration for the RBY1-M mobile base navigation environment.

    The robot drives its mecanum-wheel base to reach a target position.
    The upper body is held fixed at the default pose.
    Actions: 4 (wheel_fl, wheel_fr, wheel_rl, wheel_rr velocity)
    Observations: 2 relative target (dx, dy) + 1 heading + 4 wheel vel + 2 base linear vel = 9
    """

    # env
    decimation = 4
    episode_length_s = 300.0  # 5 minutes for teleop

    # action space: 4 mecanum wheels
    action_space = 4
    # observation space: 2 target_rel + 1 heading + 4 wheel_vel + 2 base_vel = 9
    observation_space = 9
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = RBY1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)

    # joint names
    wheel_joint_names = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]

    # action scale
    wheel_action_scale = 10.0  # [rad/s]

    # target
    target_distance_range = (2.0, 5.0)

    # reward scales
    rew_scale_distance = -1.0
    rew_scale_reached = 10.0
    rew_scale_heading = 0.5
    rew_scale_action_rate = -0.01

    # termination / success
    target_reached_threshold = 0.5
