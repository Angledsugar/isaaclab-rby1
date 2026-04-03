"""Standalone script to convert RBY1 URDF to USD."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert RBY1 URDF to USD.")
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the robot base.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import omni.kit.app
import omni.kit.commands

# Enable the URDF importer extension
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf-2.4.31", True)

URDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "source", "rby1", "rby1", "assets", "rby1a", "model.urdf")
)
USD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "source", "rby1", "rby1", "assets", "rby1a")
)

suffix = "_fixed" if args_cli.fix_base else "_mobile"
USD_OUTPUT = os.path.join(USD_DIR, f"rby1{suffix}", "model.usd")
os.makedirs(os.path.dirname(USD_OUTPUT), exist_ok=True)

print(f"[INFO] URDF: {URDF_PATH}")
print(f"[INFO] USD output: {USD_OUTPUT}")
print(f"[INFO] fix_base: {args_cli.fix_base}")

# Create import config
_, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.set_distance_scale(1.0)
import_config.set_make_default_prim(True)
import_config.set_create_physics_scene(False)
import_config.set_density(0.0)
import_config.set_convex_decomp(False)
import_config.set_collision_from_visuals(False)
import_config.set_merge_fixed_joints(False)
import_config.set_fix_base(args_cli.fix_base)
import_config.set_self_collision(False)
import_config.set_replace_cylinders_with_capsules(True)

# Parse URDF
print("\n[STEP 1] Parsing URDF...")
result, robot_model = omni.kit.commands.execute(
    "URDFParseFile", urdf_path=URDF_PATH, import_config=import_config
)
print(f"  Result: {result}, Robot: {robot_model.name}, Links: {len(robot_model.links)}, Joints: {len(robot_model.joints)}")

if result:
    # Import
    print("\n[STEP 2] Converting to USD...")
    try:
        omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=URDF_PATH,
            urdf_robot=robot_model,
            import_config=import_config,
            dest_path=USD_OUTPUT,
        )
        if os.path.exists(USD_OUTPUT):
            print(f"  SUCCESS: {USD_OUTPUT} ({os.path.getsize(USD_OUTPUT)} bytes)")
        else:
            print("  FAILED: USD file not created")
    except Exception as e:
        print(f"  FAILED: {e}")

simulation_app.close()
