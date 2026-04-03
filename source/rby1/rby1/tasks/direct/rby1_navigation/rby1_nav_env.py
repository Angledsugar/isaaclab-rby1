# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.math import sample_uniform

from .rby1_nav_env_cfg import Rby1NavEnvCfg


class Rby1NavEnv(DirectRLEnv):
    """Navigation environment for the RBY1 mobile base.

    The robot controls only its 2 wheels (differential drive) to reach a random target.
    The upper body holds its default pose.
    """

    cfg: Rby1NavEnvCfg

    def __init__(self, cfg: Rby1NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Wheel joint indices
        self._wheel_dof_idx, _ = self.robot.find_joints(self.cfg.wheel_joint_names)

        # Debug: print robot info
        all_joint_names = self.robot.joint_names
        print(f"[DEBUG] All joints ({len(all_joint_names)}): {all_joint_names}")
        print(f"[DEBUG] Wheel joint indices: {self._wheel_dof_idx}")
        print(f"[DEBUG] Wheel joint names: {[all_joint_names[i] for i in self._wheel_dof_idx]}")
        print(f"[DEBUG] Robot mass: {self.robot.data.default_mass.sum().item():.1f} kg")

        # Find all non-wheel joints to hold them fixed
        upper_body_names = [n for n in all_joint_names if n not in self.cfg.wheel_joint_names]
        if upper_body_names:
            self._upper_body_dof_idx, _ = self.robot.find_joints(upper_body_names)
            self._default_upper_body_pos = self.robot.data.default_joint_pos[:, self._upper_body_dof_idx].clone()
        else:
            self._upper_body_dof_idx = []

        # Target positions (x, y) per env in world frame
        self._target_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # Previous actions
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # Ground plane with high friction
        ground_cfg = GroundPlaneCfg(
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Wheels: velocity control
        wheel_vel_target = self.actions * self.cfg.wheel_action_scale
        self.robot.set_joint_velocity_target(wheel_vel_target, joint_ids=self._wheel_dof_idx)

        # Hold upper body at default pose
        if len(self._upper_body_dof_idx) > 0:
            self.robot.set_joint_position_target(self._default_upper_body_pos, joint_ids=self._upper_body_dof_idx)

        # Debug: print every 100 steps
        if self.episode_length_buf[0] % 100 == 0:
            actual_wheel_vel = self.robot.data.joint_vel[0, self._wheel_dof_idx]
            root_pos = self.robot.data.root_pos_w[0]
            root_vel = self.robot.data.root_lin_vel_w[0]
            print(
                f"[step {self.episode_length_buf[0].item():4d}] "
                f"action={self.actions[0].tolist()} "
                f"vel_target={wheel_vel_target[0].tolist()} "
                f"actual_vel={actual_wheel_vel.tolist()} "
                f"root_pos=[{root_pos[0]:.3f},{root_pos[1]:.3f},{root_pos[2]:.3f}] "
                f"root_vel=[{root_vel[0]:.3f},{root_vel[1]:.3f},{root_vel[2]:.3f}]"
            )

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w[:, :2]  # (N, 2) xy
        root_quat = self.robot.data.root_quat_w  # (N, 4)
        root_vel = self.robot.data.root_lin_vel_w[:, :2]  # (N, 2) vx, vy

        # Heading angle from quaternion (yaw)
        heading = self._quat_to_yaw(root_quat)  # (N,)

        # Relative target in robot local frame
        target_rel_world = self._target_pos - root_pos  # (N, 2)
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        target_local_x = cos_h * target_rel_world[:, 0] + sin_h * target_rel_world[:, 1]
        target_local_y = -sin_h * target_rel_world[:, 0] + cos_h * target_rel_world[:, 1]

        # Wheel velocities
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_dof_idx]

        obs = torch.cat(
            (
                target_local_x.unsqueeze(-1),  # 1
                target_local_y.unsqueeze(-1),  # 1
                heading.unsqueeze(-1),  # 1
                wheel_vel,  # 2
                root_vel,  # 2
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(self._target_pos - root_pos, dim=-1)

        # Heading reward: dot product of heading direction and target direction
        root_quat = self.robot.data.root_quat_w
        heading = self._quat_to_yaw(root_quat)
        target_dir = self._target_pos - root_pos
        target_angle = torch.atan2(target_dir[:, 1], target_dir[:, 0])
        heading_error = torch.abs(self._wrap_angle(target_angle - heading))
        heading_reward = self.cfg.rew_scale_heading * torch.cos(heading_error)

        # Distance reward
        distance_reward = self.cfg.rew_scale_distance * distance

        # Reached bonus
        reached = (distance < self.cfg.target_reached_threshold).float()
        reached_reward = self.cfg.rew_scale_reached * reached

        # Action rate penalty
        action_rate = self.cfg.rew_scale_action_rate * torch.sum(
            torch.square(self.actions - self._prev_actions), dim=-1
        )

        total = distance_reward + heading_reward + reached_reward + action_rate
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if robot falls over
        # gravity_proj z: -1.0 = upright, 0.0 = 90deg tilt, +1.0 = upside down
        root_quat = self.robot.data.root_quat_w
        gravity_proj = self._quat_rotate_inverse(root_quat, torch.tensor([0.0, 0.0, -1.0], device=self.device))
        fallen = gravity_proj[:, 2] > -0.17  # tilt > ~80 degrees from upright

        # Debug: print if about to terminate
        if fallen[0].item():
            print(f"[DONE] Robot fallen! gravity_z={gravity_proj[0, 2]:.3f}")
        if time_out[0].item():
            print(f"[DONE] Time out!")

        # Don't terminate on target reached (for teleop)
        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Randomize target position
        num_reset = len(env_ids)
        dist = sample_uniform(
            self.cfg.target_distance_range[0],
            self.cfg.target_distance_range[1],
            (num_reset,),
            self.device,
        )
        angle = sample_uniform(-math.pi, math.pi, (num_reset,), self.device)
        env_origins_xy = self.scene.env_origins[env_ids, :2]
        self._target_pos[env_ids, 0] = env_origins_xy[:, 0] + dist * torch.cos(angle)
        self._target_pos[env_ids, 1] = env_origins_xy[:, 1] + dist * torch.sin(angle)

        # Reset action history
        self._prev_actions[env_ids] = 0.0

    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion (w, x, y, z)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate a vector by the inverse of a quaternion (w, x, y, z)."""
        q_w = quat[:, 0:1]
        q_vec = quat[:, 1:4]
        if vec.dim() == 1:
            vec = vec.unsqueeze(0).expand(quat.shape[0], -1)
        a = vec * (2.0 * q_w**2 - 1.0)
        b = torch.cross(q_vec, vec, dim=-1) * q_w * 2.0
        c = q_vec * torch.sum(q_vec * vec, dim=-1, keepdim=True) * 2.0
        return a - b + c
