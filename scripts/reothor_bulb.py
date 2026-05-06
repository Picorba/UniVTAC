"""reauthor_bulb_assets.py — Offline USD pre-processing utility.

Run this script ONCE before any screw_bulb training to convert the Isaac Sim
light-bulb and socket assets into UIPC-compatible USD files.

Why this is necessary
---------------------
Isaac Sim assets are authored for the PhysX backend:
  • UsdPhysics.RigidBodyAPI   → PhysX rigid body activation
  • UsdPhysics.CollisionAPI   → convexHull / convexDecomposition mesh approx.
  • UsdPhysics.MassAPI        → PhysX inertia tensor override
  • PhysxSchema.PhysxRigidBodyAPI → solver-specific tuning

UIPC (tacex_uipc) manages all of the above through its own binding layer,
passed as arguments to `actor_manager.add_from_usd_file(...)`.  Leaving
PhysX schemas in the USD does not cause a crash but can produce conflicting
physics state and incorrect contact normals on the screw base cylinder.

What this script does
---------------------
1. Locate the Isaac Sim bulb and socket USDs via the ISAAC_ASSETS_PATH env var.
   Falls back to constructing minimal primitive-based USDs if the assets are
   not found (useful for CI or machines without a full Isaac installation).
2. Strip all UsdPhysics.* and PhysxSchema.* API schemas from every prim.
3. Replace mesh-based collision approximations with analytical proxies that
   UIPC handles robustly:
     • Bulb glass  → UsdGeom.Sphere   (radius  = BULB_GLASS_RADIUS)
     • Bulb base   → UsdGeom.Cylinder (radius  = BASE_RADIUS,
                                       height  = BASE_HEIGHT)
     • Socket body → UsdGeom.Cylinder (outer shell, kinematic)
   Thread geometry is intentionally OMITTED — engagement is procedural.
4. Write outputs to OUTPUT_DIR (defaults to the task asset directory).

Usage
-----
    # From the repo root, inside Isaac Sim's Python env:
    python scripts/reauthor_bulb_assets.py

    # Override paths:
    ISAAC_ASSETS_PATH=/path/to/isaac OUTPUT_DIR=assets/bulb \
        python scripts/reauthor_bulb_assets.py

Known Isaac Sim asset paths (verify against your installation version)
----------------------------------------------------------------------
Isaac 4.x (2024):
    omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props/
        → search "bulb" or "E27" in the Content browser

Isaac 2023.x:
    omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.0/Isaac/Props/

If no bulb asset is found, this script falls back to procedurally generating
a minimal USDA from UsdGeom primitives.  The visual fidelity is lower but the
collision geometry is identical.
"""

from __future__ import annotations

import os
import pathlib
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# USD Python bindings — available inside Isaac Sim's Python environment.
# Outside Isaac, install with:  pip install usd-core
# ---------------------------------------------------------------------------
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Kind

# ---------------------------------------------------------------------------
# Geometry constants  (E27 Edison-screw standard, scaled to sim units [m])
# ---------------------------------------------------------------------------
BULB_GLASS_RADIUS = 0.030   # m  — A60 / standard A-bulb glass sphere radius
BASE_RADIUS       = 0.0135  # m  — E27 base outer radius  (27 mm ⌀ / 2)
BASE_HEIGHT       = 0.028   # m  — E27 base thread length
SOCKET_INNER_R    = 0.0140  # m  — socket inner bore (slightly larger than base)
SOCKET_OUTER_R    = 0.030   # m  — socket outer shell radius
SOCKET_HEIGHT     = 0.032   # m  — socket depth

# Offset of the bulb base centre below the glass centroid
BASE_OFFSET_Z = -(BULB_GLASS_RADIUS + BASE_HEIGHT * 0.5)

# ---------------------------------------------------------------------------
# PhysX schema namespaces to strip
# ---------------------------------------------------------------------------
PHYSX_SCHEMAS = {
    "PhysicsRigidBodyAPI",
    "PhysicsCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsMaterialAPI",
    "PhysicsArticulationRootAPI",
    "PhysicsJoint",
    "PhysxRigidBodyAPI",
    "PhysxCollisionAPI",
    "PhysxJointAPI",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_physics_schemas(prim: Usd.Prim) -> None:
    """Recursively remove all UsdPhysics / Physx schemas from a prim tree."""
    applied = prim.GetAppliedSchemas()
    for schema_name in list(applied):
        # Schema names are like "PhysicsRigidBodyAPI", "PhysxCollisionAPI", …
        for prefix in ("Physics", "Physx"):
            if schema_name.startswith(prefix):
                prim.RemoveAppliedSchema(schema_name)
                break
    for child in prim.GetChildren():
        _strip_physics_schemas(child)


def _add_sphere_collision_proxy(
    stage: Usd.Stage,
    parent_path: str,
    radius: float,
    offset_z: float = 0.0,
    purpose: str = "proxy",
) -> UsdGeom.Sphere:
    proxy_path = f"{parent_path}/collision_glass"
    sphere = UsdGeom.Sphere.Define(stage, proxy_path)
    sphere.GetRadiusAttr().Set(radius)
    UsdGeom.Imageable(sphere).GetPurposeAttr().Set(purpose)
    xform_ops = sphere.AddXformOp(UsdGeom.XformOp.TypeTranslate)
    xform_ops.Set(Gf.Vec3d(0.0, 0.0, offset_z))
    return sphere


def _add_cylinder_collision_proxy(
    stage: Usd.Stage,
    parent_path: str,
    radius: float,
    height: float,
    offset_z: float = 0.0,
    name: str = "collision_base",
    purpose: str = "proxy",
) -> UsdGeom.Cylinder:
    proxy_path = f"{parent_path}/{name}"
    cyl = UsdGeom.Cylinder.Define(stage, proxy_path)
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    UsdGeom.Imageable(cyl).GetPurposeAttr().Set(purpose)
    xform_ops = cyl.AddXformOp(UsdGeom.XformOp.TypeTranslate)
    xform_ops.Set(Gf.Vec3d(0.0, 0.0, offset_z))
    return cyl


# ---------------------------------------------------------------------------
# Fallback: build a minimal bulb USD from primitives
# ---------------------------------------------------------------------------
def _build_primitive_bulb_usd(output_path: pathlib.Path) -> None:
    """
    Create a minimal bulb USD from UsdGeom primitives.

    Hierarchy
    ---------
    /bulb                  Xform  (root — driven by actor_manager)
      /glass               Sphere (visual + collision proxy, render purpose)
      /base                Cylinder (visual + collision proxy, render purpose)

    The glass and base share the render purpose so they are always visible.
    The collision proxy role is filled by the same prims — UIPC reads the
    analytical shapes directly without needing a separate collision mesh.
    """
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/bulb")
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)

    # Glass sphere
    glass = UsdGeom.Sphere.Define(stage, "/bulb/glass")
    glass.GetRadiusAttr().Set(BULB_GLASS_RADIUS)
    UsdGeom.Imageable(glass).GetPurposeAttr().Set("render")
    glass.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )

    # Screw base cylinder
    base = UsdGeom.Cylinder.Define(stage, "/bulb/base")
    base.GetRadiusAttr().Set(BASE_RADIUS)
    base.GetHeightAttr().Set(BASE_HEIGHT)
    UsdGeom.Imageable(base).GetPurposeAttr().Set("render")
    base.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(
        Gf.Vec3d(0.0, 0.0, BASE_OFFSET_Z)
    )

    stage.GetRootLayer().documentation = (
        "E27 light bulb — primitive fallback, UIPC-compatible. "
        "Thread geometry omitted; engagement is procedural."
    )
    stage.Save()
    print(f"[reauthor] primitive bulb written → {output_path}")


def _build_primitive_socket_usd(output_path: pathlib.Path) -> None:
    """
    Create a minimal socket USD.

    The socket is a static kinematic body (passed as kinematic=True to
    actor_manager).  Its collision geometry is an outer cylinder shell;
    the inner bore is left open — the procedural screw constraint handles
    radial alignment, so a physical bore is unnecessary.

    Hierarchy
    ---------
    /socket                Xform
      /outer_shell         Cylinder  (render + collision proxy)
    """
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/socket")
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)

    shell = UsdGeom.Cylinder.Define(stage, "/socket/outer_shell")
    shell.GetRadiusAttr().Set(SOCKET_OUTER_R)
    shell.GetHeightAttr().Set(SOCKET_HEIGHT)
    UsdGeom.Imageable(shell).GetPurposeAttr().Set("render")

    stage.GetRootLayer().documentation = (
        "E27 socket — primitive fallback, UIPC-compatible, kinematic static."
    )
    stage.Save()
    print(f"[reauthor] primitive socket written → {output_path}")


# ---------------------------------------------------------------------------
# Main re-authoring pipeline (Isaac source → UIPC output)
# ---------------------------------------------------------------------------
def reauthor_from_isaac(
    isaac_bulb_path:   str,
    isaac_socket_path: str,
    output_dir:        pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Load Isaac assets, strip PhysX schemas, add UIPC collision proxies,
    and save to output_dir.

    Returns (bulb_out_path, socket_out_path).
    """
    bulb_out   = output_dir / "bulb_e27.usd"
    socket_out = output_dir / "socket_e27.usd"

    for src_path, out_path, asset_name, build_fn in [
        (isaac_bulb_path,   bulb_out,   "bulb",   _patch_bulb),
        (isaac_socket_path, socket_out, "socket", _patch_socket),
    ]:
        print(f"[reauthor] processing {asset_name}: {src_path}")
        stage = Usd.Stage.Open(src_path)
        if not stage:
            raise FileNotFoundError(
                f"Could not open Isaac USD: {src_path}\n"
                "Verify ISAAC_ASSETS_PATH and check the Content browser for "
                "the exact asset path."
            )
        _strip_physics_schemas(stage.GetPseudoRoot())
        build_fn(stage)
        stage.Export(str(out_path))
        print(f"[reauthor] {asset_name} written → {out_path}")

    return bulb_out, socket_out


def _patch_bulb(stage: Usd.Stage) -> None:
    """Add UIPC-compatible collision proxies to a loaded Isaac bulb stage."""
    # Find the root prim (typically /light_bulb or /World/light_bulb)
    root_path = _find_root_xform(stage)
    _add_sphere_collision_proxy(
        stage, root_path,
        radius=BULB_GLASS_RADIUS, offset_z=0.0,
    )
    _add_cylinder_collision_proxy(
        stage, root_path,
        radius=BASE_RADIUS, height=BASE_HEIGHT,
        offset_z=BASE_OFFSET_Z, name="collision_base",
    )


def _patch_socket(stage: Usd.Stage) -> None:
    """Add UIPC-compatible collision proxy to a loaded Isaac socket stage."""
    root_path = _find_root_xform(stage)
    _add_cylinder_collision_proxy(
        stage, root_path,
        radius=SOCKET_OUTER_R, height=SOCKET_HEIGHT,
        offset_z=0.0, name="collision_shell",
    )


def _find_root_xform(stage: Usd.Stage) -> str:
    """Return the path of the first Xform child of the pseudo-root."""
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.IsA(UsdGeom.Xform):
            return prim.GetPath().pathString
    # Fallback: return pseudo-root path
    return "/"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-assets",
        default=os.environ.get(
            "ISAAC_ASSETS_PATH",
            "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props",
        ),
        help="Root path of Isaac Sim asset library (Omniverse URI or local).",
    )
    parser.add_argument(
        "--bulb-subpath",
        default="Household/LightBulb/light_bulb_A60.usd",
        help="Path to bulb USD relative to --isaac-assets.",
    )
    parser.add_argument(
        "--socket-subpath",
        default="Household/LightBulb/light_bulb_socket_E27.usd",
        help="Path to socket USD relative to --isaac-assets.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "assets/bulb"),
        help="Directory where patched USDs are written.",
    )
    parser.add_argument(
        "--fallback-primitives",
        action="store_true",
        default=False,
        help=(
            "Skip Isaac lookup and always build from UsdGeom primitives. "
            "Useful on machines without a full Isaac installation."
        ),
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bulb_out   = output_dir / "bulb_e27.usd"
    socket_out = output_dir / "socket_e27.usd"

    if args.fallback_primitives:
        print("[reauthor] --fallback-primitives: building from UsdGeom primitives.")
        _build_primitive_bulb_usd(bulb_out)
        _build_primitive_socket_usd(socket_out)
        return

    # Try Isaac first; fall back to primitives on any failure.
    isaac_bulb   = f"{args.isaac_assets}/{args.bulb_subpath}"
    isaac_socket = f"{args.isaac_assets}/{args.socket_subpath}"
    try:
        reauthor_from_isaac(isaac_bulb, isaac_socket, output_dir)
    except (FileNotFoundError, Exception) as exc:
        print(
            f"[reauthor] WARNING: Isaac asset load failed ({exc}).\n"
            "           Falling back to primitive-based USD generation."
        )
        _build_primitive_bulb_usd(bulb_out)
        _build_primitive_socket_usd(socket_out)

    # ── Print geometry constants for task file cross-check ────────────────
    print("\n── Geometry constants (paste into screw_bulb.py if changed) ──")
    print(f"  BULB_GLASS_RADIUS = {BULB_GLASS_RADIUS}")
    print(f"  BASE_RADIUS       = {BASE_RADIUS}")
    print(f"  BASE_HEIGHT       = {BASE_HEIGHT}")
    print(f"  BASE_OFFSET_Z     = {BASE_OFFSET_Z:.6f}")
    print(f"  SOCKET_INNER_R    = {SOCKET_INNER_R}")
    print(f"  SOCKET_OUTER_R    = {SOCKET_OUTER_R}")
    print(f"  SOCKET_HEIGHT     = {SOCKET_HEIGHT}")
    print("──────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()