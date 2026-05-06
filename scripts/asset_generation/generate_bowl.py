"""generate_bowl.py — CadQuery bowl asset generator for Isaac Sim / USD.

Uses a loft + boolean-cut strategy (more robust than revolve in CadQuery):
  1. Outer solid  : loft from base circle (bottom) → rim circle (top)
  2. Inner void   : loft from inner-base circle (at base_thick) → inner-rim circle (top)
  3. Bowl         : outer_solid.cut(inner_void)

Usage
-----
    python generate_bowl.py --out ./bowl.usda

    python generate_bowl.py \\
        --outer-radius 0.14 \\
        --inner-radius 0.12 \\
        --height       0.08 \\
        --base-radius  0.07 \\
        --base-thick   0.008 \\
        --scale        1.0   \\
        --tess         0.5   \\
        --out          ./bowl.usda

Z convention (matches task file)
---------------------------------
    With --center-z (default ON):
        origin = geometric centre → set BOWL_BASE_POSE z = height/2
        so the bowl bottom rests on the table surface (z ≈ 0).

Dependencies
------------
    pip install cadquery
    pip install usd-core   # or run inside Isaac Sim / Omniverse Python env
"""

import argparse
import sys


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate a USD bowl asset via CadQuery.")
    p.add_argument("--outer-radius", type=float, default=0.14,
                   help="Outer radius at the bowl rim (m). Default 0.14")
    p.add_argument("--inner-radius", type=float, default=0.12,
                   help="Inner radius at the bowl rim (m). Default 0.12")
    p.add_argument("--height",       type=float, default=0.08,
                   help="Total bowl height (m). Default 0.08")
    p.add_argument("--base-radius",  type=float, default=0.07,
                   help="Outer radius of the flat foot at the bottom (m). Default 0.07")
    p.add_argument("--base-thick",   type=float, default=0.008,
                   help="Thickness of the solid base floor (m). Default 0.008")
    p.add_argument("--scale",        type=float, default=1.0,
                   help="Uniform scale applied after meshing. Default 1.0")
    p.add_argument("--tess",         type=float, default=0.5,
                   help="Tessellation tolerance (smaller = finer). Default 0.5")
    p.add_argument("--out",          type=str,   default="bowl.usda",
                   help="Output USD file path. Default ./bowl.usda")
    p.add_argument("--center-z",     dest="center_z", action="store_true",  default=True,
                   help="Origin at geometric centre (default ON)")
    p.add_argument("--no-center-z",  dest="center_z", action="store_false",
                   help="Keep origin at bottom (z=0 = table surface)")
    return p.parse_args()


# ── Geometry ──────────────────────────────────────────────────────────────────

def build_bowl(
    outer_radius: float = 0.14,
    inner_radius: float = 0.12,
    height:       float = 0.08,
    base_radius:  float = 0.07,
    base_thick:   float = 0.008,
):
    """
    Build a bowl using loft + boolean cut.

    Cross-section (not to scale):

      rim: <──outer_radius──>
           ┌────────────────┐  <- height
           │  <inner_radius>│
           │  ┌──────────┐  │
           │  │  (void)  │  │  <- walls (outer_radius - inner_radius thick)
           │  │          │  │
           │  └──────────┘  │  <- base_thick
           └────────────────┘  <- z = 0
      foot:  <─base_radius─>

    Both lofts are circles at two heights → clean frustum walls.
    The inner loft starts at z = base_thick, leaving a solid floor.
    """
    try:
        import cadquery as cq
    except ImportError:
        sys.exit(
            "CadQuery not found.\n"
            "Install: pip install cadquery\n"
            "Or activate your Omniverse / Isaac Sim Python environment."
        )

    wall_thick = outer_radius - inner_radius

    # Inner base radius: shrink by wall thickness, but never go negative
    inner_base_radius = max(base_radius - wall_thick, base_radius * 0.2)

    # ── 1. Outer solid ────────────────────────────────────────────────────────
    # Loft: base_radius circle at z=0  →  outer_radius circle at z=height
    outer_solid = (
        cq.Workplane("XY")
        .circle(base_radius)
        .workplane(offset=height)
        .circle(outer_radius)
        .loft()
    )

    # ── 2. Inner void ─────────────────────────────────────────────────────────
    # Loft: inner_base_radius circle at z=base_thick  →  inner_radius at z=height
    # Starting above z=0 preserves the solid base floor.
    inner_void = (
        cq.Workplane("XY")
        .workplane(offset=base_thick)
        .circle(inner_base_radius)
        .workplane(offset=height - base_thick)
        .circle(inner_radius)
        .loft()
    )

    # ── 3. Boolean cut ────────────────────────────────────────────────────────
    bowl = outer_solid.cut(inner_void)

    return bowl


# ── USD export ────────────────────────────────────────────────────────────────

def export_to_usd(
    shape,
    out_path:  str,
    scale:     float = 1.0,
    center_z:  bool  = True,
    tess:      float = 0.5,
):
    """Tessellate the CadQuery solid and write a USD file."""
    try:
        from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, Vt
    except ImportError:
        sys.exit(
            "pxr (USD) not found.\n"
            "Install: pip install usd-core\n"
            "Or run inside your Isaac Sim Python environment."
        )

    # ── Tessellate ────────────────────────────────────────────────────────────
    verts_raw, tris_raw = shape.val().tessellate(tess)

    if not verts_raw:
        sys.exit(
            "Tessellation produced 0 vertices — "
            "check geometry parameters (inner-radius must be < outer-radius)."
        )

    print(f"  Tessellation: {len(verts_raw)} vertices, {len(tris_raw)} triangles")

    # Scale and optionally Z-centre
    points = [(v.x * scale, v.y * scale, v.z * scale) for v in verts_raw]

    if center_z:
        zs    = [p[2] for p in points]
        z_mid = (min(zs) + max(zs)) / 2.0
        points = [(p[0], p[1], p[2] - z_mid) for p in points]

    face_counts  = [3] * len(tris_raw)
    face_indices = []
    for tri in tris_raw:
        face_indices.extend([tri[0], tri[1], tri[2]])

    # ── USD stage ─────────────────────────────────────────────────────────────
    stage = Usd.Stage.CreateNew(out_path)
    stage.SetMetadata("upAxis",        "Z")
    stage.SetMetadata("metersPerUnit", 1.0)

    root = UsdGeom.Xform.Define(stage, "/Bowl")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/Bowl/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    # ── Material: warm ceramic off-white ──────────────────────────────────────
    mat = UsdShade.Material.Define(stage, "/Bowl/Materials/BowlMat")
    sh  = UsdShade.Shader.Define(stage, "/Bowl/Materials/BowlMat/PBRShader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92, 0.88, 0.80))
    sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.35)
    sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(mat)

    # ── Physics hints ─────────────────────────────────────────────────────────
    try:
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()) \
            .GetApproximationAttr().Set("convexDecomposition")
    except Exception:
        pass  # UsdPhysics not available here — Isaac Sim handles it at import

    stage.Save()
    print(f"  Saved -> {out_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(args):
    h     = args.height       * args.scale
    r_out = args.outer_radius * args.scale
    r_in  = args.inner_radius * args.scale
    wall  = (args.outer_radius - args.inner_radius) * args.scale

    print()
    print("── Bowl generation complete ──────────────────────────────────────")
    print(f"  Output          : {args.out}")
    print(f"  Outer diameter  : {r_out*2*100:.1f} cm   (rim)")
    print(f"  Inner diameter  : {r_in*2*100:.1f} cm   (opening)")
    print(f"  Height          : {h*100:.1f} cm")
    print(f"  Wall thickness  : {wall*100:.2f} cm")
    print(f"  Base thickness  : {args.base_thick*args.scale*100:.2f} cm")
    print(f"  Z-centred       : {args.center_z}")
    print()
    print("  ── Paste into your task file ─────────────────────────────────")
    print(f"  BOWL_BASE_POSE = Pose([x, y, {h/2:.4f}], [1, 0, 0, 0])")
    print(f"  # z = {h/2:.4f} m  →  bowl bottom sits on table (z ≈ 0)")
    print("──────────────────────────────────────────────────────────────────")
    print()


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.inner_radius >= args.outer_radius:
        sys.exit("ERROR: --inner-radius must be strictly less than --outer-radius")
    if args.base_radius >= args.outer_radius:
        sys.exit("ERROR: --base-radius must be strictly less than --outer-radius")
    if args.base_thick >= args.height:
        sys.exit("ERROR: --base-thick must be strictly less than --height")

    print("[generate_bowl] Building geometry ...")
    bowl = build_bowl(
        outer_radius = args.outer_radius,
        inner_radius = args.inner_radius,
        height       = args.height,
        base_radius  = args.base_radius,
        base_thick   = args.base_thick,
    )

    print("[generate_bowl] Exporting USD ...")
    export_to_usd(
        shape    = bowl,
        out_path = args.out,
        scale    = args.scale,
        center_z = args.center_z,
        tess     = args.tess,
    )

    print_summary(args)


if __name__ == "__main__":
    main()