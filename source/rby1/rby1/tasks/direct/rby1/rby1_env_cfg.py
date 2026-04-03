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
class Rby1EnvCfg(DirectRLEnvCfg):
    """Configuration for the RBY1-M whole-body control environment.

    Actions: 4 mecanum wheels + 6 torso + 7 right arm + 7 left arm = 24 DOF
    Observations: 24 joint pos + 24 joint vel + 3 projected gravity = 51
    """

    # env
    decimation = 4
    episode_length_s = 10.0

    # action space: 4 wheels + 6 torso + 7 right arm + 7 left arm = 24 DOF
    action_space = 24
    # observation space: 24 joint pos + 24 joint vel + 3 projected gravity = 51
    observation_space = 51
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = RBY1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # joint groups
    wheel_joint_names = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
    torso_joint_names = ["torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5"]
    right_arm_joint_names = [
        "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
        "right_arm_4", "right_arm_5", "right_arm_6",
    ]
    left_arm_joint_names = [
        "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
        "left_arm_4", "left_arm_5", "left_arm_6",
    ]

    # action scale
    wheel_action_scale = 10.0  # [rad/s] - velocity control for wheels
    upper_body_action_scale = 0.5  # [rad] - delta position control for upper body

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    rew_scale_joint_pos = -0.1
    rew_scale_joint_vel = -0.01
    rew_scale_joint_acc = -0.0001
    rew_scale_action_rate = -0.01

    # termination
    max_torso_tilt = 1.0

    # reset
    initial_joint_pos_noise = 0.1
