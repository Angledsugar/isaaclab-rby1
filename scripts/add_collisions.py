"""Add CollisionAPI to all visual meshes in the RBY1-M base USD layer.

This uses the actual visual mesh geometry as collision, so collision shapes
exactly match the visual appearance of the robot.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Add collisions to RBY1-M USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

from pxr import Usd, UsdPhysics

BASE_USD = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "source",
        "rby1",
        "rby1",
        "assets",
        "rby1m",
        "model",
        "configuration",
        "model_base.usd",
    )
)

print(f"[INFO] Editing: {BASE_USD}")
stage = Usd.Stage.Open(BASE_USD)

# Find all Mesh prims under /visuals and apply CollisionAPI
count = 0
for prim in stage.Traverse():
    if prim.GetTypeName() == "Mesh":
        path = str(prim.GetPath())
        # Apply CollisionAPI
        UsdPhysics.CollisionAPI.Apply(prim)
        # Set mesh collision approximation to convexHull for performance
        if hasattr(UsdPhysics, "MeshCollisionAPI"):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            prim.GetAttribute("physics:approximation").Set("convexHull")
        link_name = path.split("/")[2] if len(path.split("/")) > 2 else "?"
        count += 1
        print(f"  [OK] {link_name}: {path}")

stage.GetRootLayer().Save()
print(f"\n[DONE] Applied CollisionAPI to {count} visual meshes")

# Verify via composed stage (model.usd)
MODEL_USD = os.path.join(os.path.dirname(BASE_USD), "..", "model.usd")
stage2 = Usd.Stage.Open(MODEL_USD)
verify = sum(1 for p in stage2.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
print(f"[VERIFY] {verify} prims with CollisionAPI in composed stage")

simulation_app.close()
