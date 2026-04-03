# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .rby1_env_cfg import Rby1EnvCfg


class Rby1Env(DirectRLEnv):
    """Direct RL environment for the RBY1 whole-body control task.

    The robot controls wheels (velocity) and upper body (delta position).
    """

    cfg: Rby1EnvCfg

    def __init__(self, cfg: Rby1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve joint indices
        wheel_names = self.cfg.wheel_joint_names
        upper_body_names = (
            self.cfg.torso_joint_names + self.cfg.right_arm_joint_names + self.cfg.left_arm_joint_names
        )
        all_controlled = wheel_names + upper_body_names

        self._wheel_dof_idx, _ = self.robot.find_joints(wheel_names)
        self._upper_body_dof_idx, _ = self.robot.find_joints(upper_body_names)
        self._controlled_dof_idx, _ = self.robot.find_joints(all_controlled)

        self._num_wheels = len(self._wheel_dof_idx)
        self._num_upper_body = len(self._upper_body_dof_idx)

        # Cache references
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Default joint positions for upper body (reward target)
        self._default_upper_body_pos = self.robot.data.default_joint_pos[:, self._upper_body_dof_idx].clone()

        # Previous actions for action rate penalty
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_joint_vel = torch.zeros(self.num_envs, len(self._controlled_dof_idx), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Split actions: [wheels (2), upper_body (20)]
        wheel_actions = self.actions[:, : self._num_wheels]
        upper_body_actions = self.actions[:, self._num_wheels :]

        # Wheels: velocity control
        wheel_vel_target = wheel_actions * self.cfg.wheel_action_scale
        self.robot.set_joint_velocity_target(wheel_vel_target, joint_ids=self._wheel_dof_idx)

        # Upper body: delta position control
        current_pos = self.joint_pos[:, self._upper_body_dof_idx]
        targets = current_pos + upper_body_actions * self.cfg.upper_body_action_scale
        self.robot.set_joint_position_target(targets, joint_ids=self._upper_body_dof_idx)

        # Debug print (env 0 only, every 50 steps)
        if self.episode_length_buf[0] % 50 == 0:
            actual_wheel_vel = self.joint_vel[0, self._wheel_dof_idx]
            root_pos = self.robot.data.root_pos_w[0]
            root_vel = self.robot.data.root_lin_vel_w[0]
            print(
                f"[step {self.episode_length_buf[0].item():4d}] "
                f"wheel_target=[{wheel_vel_target[0,0]:.2f}, {wheel_vel_target[0,1]:.2f}] "
                f"wheel_actual=[{actual_wheel_vel[0]:.2f}, {actual_wheel_vel[1]:.2f}] "
                f"root_pos=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}] "
                f"root_vel=[{root_vel[0]:.2f}, {root_vel[1]:.2f}, {root_vel[2]:.2f}]"
            )

    def _get_observations(self) -> dict:
        # Projected gravity in robot base frame
        root_quat = self.robot.data.root_quat_w  # (N, 4)
        gravity_proj = self._quat_rotate_inverse(root_quat, torch.tensor([0.0, 0.0, -1.0], device=self.device))

        obs = torch.cat(
            (
                self.joint_pos[:, self._controlled_dof_idx],  # 22
                self.joint_vel[:, self._controlled_dof_idx],  # 22
                gravity_proj,  # 3
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_joint_pos,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_joint_acc,
            self.cfg.rew_scale_action_rate,
            self.joint_pos[:, self._upper_body_dof_idx],
            self.joint_vel[:, self._controlled_dof_idx],
            self._default_upper_body_pos,
            self._prev_joint_vel,
            self.actions,
            self._prev_actions,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check torso tilt using projected gravity
        root_quat = self.robot.data.root_quat_w
        gravity_proj = self._quat_rotate_inverse(root_quat, torch.tensor([0.0, 0.0, -1.0], device=self.device))
        torso_tilted = gravity_proj[:, 2] < torch.cos(torch.tensor(self.cfg.max_torso_tilt, device=self.device))

        return torso_tilted, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joint positions with noise (only upper body, keep wheels at 0)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._upper_body_dof_idx] += sample_uniform(
            -self.cfg.initial_joint_pos_noise,
            self.cfg.initial_joint_pos_noise,
            joint_pos[:, self._upper_body_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset action history
        self._prev_actions[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0

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


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_joint_pos: float,
    rew_scale_joint_vel: float,
    rew_scale_joint_acc: float,
    rew_scale_action_rate: float,
    upper_body_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    default_upper_body_pos: torch.Tensor,
    prev_joint_vel: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    # Alive bonus
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # Termination penalty
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # Upper body position deviation from default
    rew_joint_pos = rew_scale_joint_pos * torch.sum(torch.square(upper_body_pos - default_upper_body_pos), dim=-1)
    # Joint velocity penalty (all controlled joints)
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1)
    # Joint acceleration penalty (smoothness)
    joint_acc = joint_vel - prev_joint_vel
    rew_joint_acc = rew_scale_joint_acc * torch.sum(torch.square(joint_acc), dim=-1)
    # Action rate penalty (smoothness)
    rew_action_rate = rew_scale_action_rate * torch.sum(torch.square(actions - prev_actions), dim=-1)

    total_reward = rew_alive + rew_termination + rew_joint_pos + rew_joint_vel + rew_joint_acc + rew_action_rate
    return total_reward
