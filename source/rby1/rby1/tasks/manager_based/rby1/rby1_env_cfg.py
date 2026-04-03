# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from rby1.assets import RBY1_CFG  # isort:skip


##
# Scene definition
##


@configclass
class Rby1SceneCfg(InteractiveSceneCfg):
    """Configuration for the RBY1 upper-body scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = RBY1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

# All upper-body joint names for convenience
_TORSO_JOINTS = ["torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5"]
_RIGHT_ARM_JOINTS = ["right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6"]
_LEFT_ARM_JOINTS = ["left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6"]
_UPPER_BODY_JOINTS = _TORSO_JOINTS + _RIGHT_ARM_JOINTS + _LEFT_ARM_JOINTS


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_UPPER_BODY_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_upper_body = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_UPPER_BODY_JOINTS),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    # (3) Keep joints near default position
    joint_pos_default = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_UPPER_BODY_JOINTS), "target": 0.0},
    )
    # (4) Minimize joint velocities
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_UPPER_BODY_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class Rby1EnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment configuration for RBY1."""

    # Scene settings
    scene: Rby1SceneCfg = Rby1SceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10
        self.viewer.eye = (4.0, 0.0, 3.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
