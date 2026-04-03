# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rainbow Robotics RBY1 Model M (mecanum wheel) robot."""

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

_RBY1M_ASSET_DIR = os.path.join(os.path.dirname(__file__), "rby1m")
_RBY1M_URDF_PATH = os.path.join(_RBY1M_ASSET_DIR, "model.urdf")

##
# Configuration - RBY1 Model M: mecanum-wheel mobile humanoid
# Total DOF: 4 wheels + 6 torso + 14 arms + 2 head + 4 grippers = 30
##

RBY1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_RBY1M_URDF_PATH,
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=False,
        collision_from_visuals=True,
        collider_type="convex_decomposition",
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type={
                "wheel_.*": "velocity",
                ".*": "position",
            },
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    "wheel_.*": 0.0,
                    ".*": 100.0,
                },
                damping={
                    "wheel_.*": 10.0,
                    ".*": 10.0,
                },
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Mecanum wheels
            "wheel_fl": 0.0,
            "wheel_fr": 0.0,
            "wheel_rl": 0.0,
            "wheel_rr": 0.0,
            # Torso
            "torso_0": 0.0,
            "torso_1": 0.0,
            "torso_2": 0.0,
            "torso_3": 0.0,
            "torso_4": 0.0,
            "torso_5": 0.0,
            # Right arm
            "right_arm_0": 0.0,
            "right_arm_1": -0.5,
            "right_arm_2": 0.0,
            "right_arm_3": -1.0,
            "right_arm_4": 0.0,
            "right_arm_5": 0.0,
            "right_arm_6": 0.0,
            # Left arm (mirrored)
            "left_arm_0": 0.0,
            "left_arm_1": 0.5,
            "left_arm_2": 0.0,
            "left_arm_3": -1.0,
            "left_arm_4": 0.0,
            "left_arm_5": 0.0,
            "left_arm_6": 0.0,
            # Head
            "head_0": 0.0,
            "head_1": 0.0,
            # Grippers
            "gripper_finger_r1": 0.0,
            "gripper_finger_r2": 0.0,
            "gripper_finger_l1": 0.0,
            "gripper_finger_l2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_[0-5]"],
            effort_limit={
                "torso_0": 500.0,
                "torso_1": 500.0,
                "torso_2": 500.0,
                "torso_3": 300.0,
                "torso_4": 300.0,
                "torso_5": 300.0,
            },
            stiffness={
                "torso_0": 800.0,
                "torso_1": 800.0,
                "torso_2": 800.0,
                "torso_3": 500.0,
                "torso_4": 500.0,
                "torso_5": 500.0,
            },
            damping={
                "torso_0": 80.0,
                "torso_1": 80.0,
                "torso_2": 80.0,
                "torso_3": 50.0,
                "torso_4": 50.0,
                "torso_5": 50.0,
            },
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_[0-6]"],
            effort_limit={
                "right_arm_0": 150.0,
                "right_arm_1": 150.0,
                "right_arm_2": 150.0,
                "right_arm_3": 80.0,
                "right_arm_4": 20.0,
                "right_arm_5": 20.0,
                "right_arm_6": 16.0,
            },
            stiffness={
                "right_arm_0": 300.0,
                "right_arm_1": 300.0,
                "right_arm_2": 300.0,
                "right_arm_3": 200.0,
                "right_arm_4": 100.0,
                "right_arm_5": 100.0,
                "right_arm_6": 80.0,
            },
            damping={
                "right_arm_0": 30.0,
                "right_arm_1": 30.0,
                "right_arm_2": 30.0,
                "right_arm_3": 20.0,
                "right_arm_4": 10.0,
                "right_arm_5": 10.0,
                "right_arm_6": 8.0,
            },
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_[0-6]"],
            effort_limit={
                "left_arm_0": 150.0,
                "left_arm_1": 150.0,
                "left_arm_2": 150.0,
                "left_arm_3": 80.0,
                "left_arm_4": 20.0,
                "left_arm_5": 20.0,
                "left_arm_6": 16.0,
            },
            stiffness={
                "left_arm_0": 300.0,
                "left_arm_1": 300.0,
                "left_arm_2": 300.0,
                "left_arm_3": 200.0,
                "left_arm_4": 100.0,
                "left_arm_5": 100.0,
                "left_arm_6": 80.0,
            },
            damping={
                "left_arm_0": 30.0,
                "left_arm_1": 30.0,
                "left_arm_2": 30.0,
                "left_arm_3": 20.0,
                "left_arm_4": 10.0,
                "left_arm_5": 10.0,
                "left_arm_6": 8.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_[0-1]"],
            effort_limit=10.0,
            stiffness=20.0,
            damping=2.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"],
            effort_limit=200.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=50.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["gripper_finger_.*"],
            effort_limit=100.0,
            stiffness=100.0,
            damping=5.0,
        ),
    },
)
"""Configuration for the Rainbow Robotics RBY1 Model M mecanum-wheel mobile humanoid."""

# Backward compatibility
RBY1_MINIMAL_CFG = RBY1_CFG
RBY1_MOBILE_CFG = RBY1_CFG
