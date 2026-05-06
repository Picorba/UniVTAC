#!/usr/bin/env python3
"""
Generate shape sorter USD assets for Isaac Sim / Isaac Lab.

Produces 6 USD files (no intermediate STL):
  cube.usd, cylinder.usd, triangular_prism.usd, star.usd, moon.usd
  shape_sorter.usd  — single solid block, 5 blind holes + clearance

Requirements:
  pip install cadquery
  pip install usd-core          # standalone use outside Isaac Sim
  # (pxr is already available inside Isaac Sim's Python environment)

Usage:
  python generate_shape_sorter.py
  python generate_shape_sorter.py --clearance 1.0 --scale 0.001 --out ./usd_shapes
"""

import argparse
import math
from pathlib import Path

import cadquery as cq

try:
    from pxr import Gf, Usd, UsdGeom, Vt
except ImportError as e:
    raise ImportError(
        "OpenUSD Python bindings not found.\n"
        "Outside Isaac Sim, install with:  pip install usd-core"
    ) from e


# ══════════════════════════════════════════════════════════════════════════════
# Default parameters  (all dimensions in mm, scaled to metres on USD export)
# ══════════════════════════════════════════════════════════════════════════════

CLEARANCE    = 0.8   # uniform gap between shape piece and sorter hole wall
SCALE        = 0.001 # mm → metres  (1 unit in USD = 1 metre)
TESSELLATION = 0.1   # OCC linear tolerance in mm  (smaller = finer mesh)
ANGULAR_TOL  = 0.1   # OCC angular tolerance in radians

# Shape piece dimensions
SHAPE_H      = 30    # height of each piece (tall enough to grasp)
CUBE_S       = 22    # cube side length
CYL_R        = 11    # cylinder radius
TRI_S        = 24    # equilateral triangle edge length
STAR_RO      = 13    # star outer (tip) radius
STAR_RI      =  5.5  # star inner (valley) radius
MOON_RO      = 13    # moon outer circle radius
MOON_RI      = 11    # moon cut-circle radius  (controls crescent thickness)
MOON_DX      =  7    # X-offset of cut circle  ← larger = thinner crescent

# Sorter block dimensions
SORTER_W     = 200   # total width
SORTER_D     =  60   # depth (front-to-back)
SORTER_H     =  40   # height
HOLE_DEPTH   =  34   # blind hole depth  →  floor = SORTER_H - HOLE_DEPTH = 6 mm
MARGIN       =  22   # from box edge to first / last hole centre
HOLE_SPACING =  39   # centre-to-centre distance between adjacent holes
# Hole centres along X: [-78, -39, 0, +39, +78]  for the defaults above


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def star_pts(r_outer: float, r_inner: float, n: int = 5):
    """
    2n vertices of an n-pointed star, first tip pointing up (+Y).
    Winding is counter-clockwise.
    """
    pts = []
    for i in range(2 * n):
        angle = math.radians(90.0 + i * 180.0 / n)
        r = r_outer if (i % 2 == 0) else r_inner
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


def tri_pts(side: float, clearance: float = 0.0):
    """
    Vertices of an equilateral triangle centred at the centroid, tip up.

    Correct formula for a uniform edge offset on an equilateral triangle:
        new_side = side + 2 * sqrt(3) * clearance
    Winding: counter-clockwise  (top → bottom-right → bottom-left).
    """
    s = side + 2.0 * math.sqrt(3) * clearance
    h = s * math.sqrt(3) / 2.0
    return [(0.0, 2 * h / 3), (s / 2, -h / 3), (-s / 2, -h / 3)]


# ══════════════════════════════════════════════════════════════════════════════
# Shape pieces — all centred at world origin (mm)
# ══════════════════════════════════════════════════════════════════════════════

def make_cube(s=CUBE_S, h=SHAPE_H) -> cq.Workplane:
    return cq.Workplane("XY").box(s, s, h)


def make_cylinder(r=CYL_R, h=SHAPE_H) -> cq.Workplane:
    return cq.Workplane("XY").cylinder(h, r)


def make_triangular_prism(side=TRI_S, h=SHAPE_H) -> cq.Workplane:
    pts = tri_pts(side)
    return (
        cq.Workplane("XY")
        .polyline(pts).close()
        .extrude(h)
        .translate((0, 0, -h / 2))
    )


def make_star(ro=STAR_RO, ri=STAR_RI, h=SHAPE_H) -> cq.Workplane:
    pts = star_pts(ro, ri)
    return (
        cq.Workplane("XY")
        .polyline(pts).close()
        .extrude(h)
        .translate((0, 0, -h / 2))
    )


def make_moon(ro=MOON_RO, ri=MOON_RI, dx=MOON_DX, h=SHAPE_H) -> cq.Workplane:
    """
    Crescent = full circle (radius ro)
             minus offset circle (radius ri, centre shifted +dx along X).
    """
    base   = cq.Workplane("XY").circle(ro).extrude(h)
    cutter = cq.Workplane("XY").center(dx, 0).circle(ri).extrude(h)
    return base.cut(cutter).translate((0, 0, -h / 2))


# ══════════════════════════════════════════════════════════════════════════════
# Shape sorter — single solid block with 5 blind holes from the top face
# ══════════════════════════════════════════════════════════════════════════════

def make_sorter(clearance: float = CLEARANCE) -> cq.Workplane:
    """
    Solid box (SORTER_W × SORTER_D × SORTER_H) with five blind holes.

    Hole order along X:  cube | cylinder | triangle | star | moon
    Holes go from the top face (Z = +SORTER_H/2) downward by HOLE_DEPTH.
    Floor thickness = SORTER_H − HOLE_DEPTH.

    Clearance strategy
    ──────────────────
    cube      : each face expanded by clearance
    cylinder  : radius expanded by clearance/2  (uniform radial gap)
    triangle  : each edge shifted outward using the exact equilateral formula
    star      : outer tips expanded; inner valleys slightly widened
    moon      : outer circle expanded; inner cut-circle contracted
    """
    c = clearance

    # Hole centre X positions (sorter box centred at X = 0)
    xs = [-SORTER_W / 2 + MARGIN + i * HOLE_SPACING for i in range(5)]
    print("SORTER CENTER", xs)
    # Z reference values
    # ─────────────────
    # Sorter box is Z-centred: Z ∈ [−SORTER_H/2, +SORTER_H/2]
    # All holes open from the top face (Z = +SORTER_H/2) downward.
    #
    # cq.box / cq.cylinder are centred solids:
    #   set their centre at z_cen → top edge lands at +SORTER_H/2
    z_cen = SORTER_H / 2 - HOLE_DEPTH / 2
    #
    # polyline.extrude(d) produces Z ∈ [0, d]:
    #   translate base to z_bot → top edge lands at +SORTER_H/2
    z_bot = SORTER_H / 2 - HOLE_DEPTH

    sorter = cq.Workplane("XY").box(SORTER_W, SORTER_D, SORTER_H)

    # ── [0] Cube hole ─────────────────────────────────────────────────────────
    s = CUBE_S + c
    sorter = sorter.cut(
        cq.Workplane("XY")
        .box(s, s, HOLE_DEPTH)
        .translate((xs[0], 0, z_cen))
    )

    # ── [1] Cylinder hole ─────────────────────────────────────────────────────
    sorter = sorter.cut(
        cq.Workplane("XY")
        .cylinder(HOLE_DEPTH, CYL_R + c / 2)
        .translate((xs[1], 0, z_cen))
    )

    # ── [2] Triangular prism hole ─────────────────────────────────────────────
    pts = tri_pts(TRI_S, clearance=c / 2)
    sorter = sorter.cut(
        cq.Workplane("XY")
        .polyline(pts).close()
        .extrude(HOLE_DEPTH)
        .translate((xs[2], 0, z_bot))
    )

    # ── [3] Star hole ─────────────────────────────────────────────────────────
    pts = star_pts(STAR_RO + c / 2, STAR_RI + c / 2)
    sorter = sorter.cut(
        cq.Workplane("XY")
        .polyline(pts).close()
        .extrude(HOLE_DEPTH)
        .translate((xs[3], 0, z_bot))
    )

    # ── [4] Moon hole ─────────────────────────────────────────────────────────
    moon_base = (
        cq.Workplane("XY")
        .circle(MOON_RO + c / 2)
        .extrude(HOLE_DEPTH)
    )
    moon_cutter = (
        cq.Workplane("XY")
        .center(MOON_DX, 0)
        .circle(MOON_RI - c / 2)
        .extrude(HOLE_DEPTH)
    )
    sorter = sorter.cut(
        moon_base.cut(moon_cutter).translate((xs[4], 0, z_bot))
    )

    return sorter


# ══════════════════════════════════════════════════════════════════════════════
# Shape sorter — lightweight collision proxy
# ══════════════════════════════════════════════════════════════════════════════

def make_sorter_collision(clearance: float = CLEARANCE) -> cq.Workplane:
    """
    Box-wall approximation of the sorter for cheap UIPC contact detection.

    Instead of the exact holed geometry (many triangles from boolean cuts),
    this builds the same volume from simple box primitives:

      • Floor slab  — solid base, height = SORTER_H - HOLE_DEPTH
      • Front/back walls  — two long boxes spanning the full width
      • End walls + inter-hole separators  — six vertical dividers

    The result has O(100) triangles vs O(10 000+) for the exact mesh, so
    UIPC contact detection is ~100× cheaper.  The visual sorter (fine mesh)
    is spawned separately as a purely-rendered prim.

    Hole order along X:  cube | cylinder | triangle | star | moon
    (same as make_sorter — hole centres at the same X positions)
    """
    c = clearance

    # Hole centre X positions — same formula as make_sorter
    xs = [-SORTER_W / 2 + MARGIN + i * HOLE_SPACING for i in range(5)]

    # Wall thickness: half the gap between adjacent hole centres minus the
    # largest inscribed-circle radius of any shape (≈ CYL_R + clearance).
    # We use a fixed value that keeps the walls non-zero for all shapes.
    wall_t = max(4.0, (HOLE_SPACING - (CUBE_S + c)) / 2)

    # ── Floor slab (the solid base below all holes) ───────────────────────────
    floor_h = SORTER_H - HOLE_DEPTH          # = 6 mm by default
    floor_z = -SORTER_H / 2 + floor_h / 2   # centre of slab in Z-centred frame
    result = cq.Workplane("XY").box(SORTER_W, SORTER_D, floor_h).translate((0, 0, floor_z))

    hole_z_cen = SORTER_H / 2 - HOLE_DEPTH / 2   # Z-centre of the hole region

    # ── Front and back walls (run along full X width) ─────────────────────────
    for sign in (-1, +1):
        y_cen = sign * (SORTER_D / 2 - wall_t / 2)
        result = result.union(
            cq.Workplane("XY")
            .box(SORTER_W, wall_t, HOLE_DEPTH)
            .translate((0, y_cen, hole_z_cen))
        )

    # ── Vertical dividers: 2 outer end walls + 4 inter-hole separators ────────
    # Divider X centres: left outer edge, midpoints between adjacent holes,
    # right outer edge.
    divider_xs = (
        [-SORTER_W / 2 + wall_t / 2]
        + [(xs[i] + xs[i + 1]) / 2 for i in range(4)]
        + [SORTER_W / 2 - wall_t / 2]
    )
    for x_cen in divider_xs:
        result = result.union(
            cq.Workplane("XY")
            .box(wall_t, SORTER_D, HOLE_DEPTH)
            .translate((x_cen, 0, hole_z_cen))
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# USD export  —  cadquery solid → pxr UsdGeom.Mesh
# ══════════════════════════════════════════════════════════════════════════════

def export_usd(
    shape: cq.Workplane,
    name: str,
    out_dir: Path,
    scale: float = SCALE,
    tess_tol: float = TESSELLATION,
    ang_tol: float = ANGULAR_TOL,
):
    """
    Tessellate a CadQuery solid and write it as a USD Mesh asset.

    USD structure
    ─────────────
    /World          (Xform, defaultPrim)
      /{name}       (Mesh)

    Parameters
    ──────────
    scale     : unit conversion applied to every vertex coordinate.
                0.001 → mm to metres (Isaac Sim standard).
    tess_tol  : OCC linear tessellation tolerance (mm before scaling).
    ang_tol   : OCC angular tessellation tolerance (radians).
    """
    # ── 1. Tessellate ─────────────────────────────────────────────────────────
    solid = shape.val()
    raw_verts, raw_tris = solid.tessellate(tess_tol, ang_tol)

    # ── 2. Build USD arrays ───────────────────────────────────────────────────
    pts = Vt.Vec3fArray(
        [Gf.Vec3f(v.x * scale, v.y * scale, v.z * scale) for v in raw_verts]
    )
    face_counts  = Vt.IntArray([3] * len(raw_tris))
    face_indices = Vt.IntArray([idx for tri in raw_tris for idx in tri])

    # Compute per-face normals (one normal per triangle vertex = flat shading)
    normals_list = []
    for tri in raw_tris:
        v0, v1, v2 = raw_verts[tri[0]], raw_verts[tri[1]], raw_verts[tri[2]]
        e1 = cq.Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
        e2 = cq.Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
        n  = e1.cross(e2).normalized()
        fn = Gf.Vec3f(n.x, n.y, n.z)
        normals_list += [fn, fn, fn]   # same normal for each vertex of this face
    normals = Vt.Vec3fArray(normals_list)

    # Axis-aligned bounding box for the Extent attribute
    xs_v = [v.x * scale for v in raw_verts]
    ys_v = [v.y * scale for v in raw_verts]
    zs_v = [v.z * scale for v in raw_verts]
    extent = Vt.Vec3fArray([
        Gf.Vec3f(min(xs_v), min(ys_v), min(zs_v)),
        Gf.Vec3f(max(xs_v), max(ys_v), max(zs_v)),
    ])

    # ── 3. Create USD stage ───────────────────────────────────────────────────
    path = str(out_dir / f"{name}.usd")
    stage = Usd.Stage.CreateNew(path)

    # Stage-level metadata
    stage.SetMetadata("metersPerUnit", 1.0)        # vertices already in metres
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Root Xform
    world_xform = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world_xform.GetPrim())

    # Mesh prim
    mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh.GetNormalsAttr().Set(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.GetExtentAttr().Set(extent)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)   # no subdivision

    stage.GetRootLayer().Save()
    print(f"  ✓  {name}.usd   ({len(raw_verts):>6} verts, {len(raw_tris):>6} tris)")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(clearance: float, scale: float):
    half_w = SORTER_W / 2
    xs = [-half_w + MARGIN + i * HOLE_SPACING for i in range(5)]
    labels = ["cube", "cylinder", "triangle", "star", "moon"]
    print(f"\n  Sorter block   : {SORTER_W} × {SORTER_D} × {SORTER_H} mm")
    print(f"  Hole depth     : {HOLE_DEPTH} mm   floor : {SORTER_H - HOLE_DEPTH} mm")
    print(f"  Clearance      : {clearance} mm")
    print(f"  Scale factor   : {scale}  ({1/scale:.0f} units = 1 metre)")
    print(f"  Hole centres X : {[round(x, 1) for x in xs]} mm")
    for label, x in zip(labels, xs):
        print(f"    {label:<14} @ X = {x:+6.1f} mm")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate shape sorter USD assets for Isaac Sim / Isaac Lab"
    )
    parser.add_argument(
        "--clearance", type=float, default=CLEARANCE,
        help=f"Hole clearance in mm (default: {CLEARANCE})",
    )
    parser.add_argument(
        "--scale", type=float, default=SCALE,
        help=f"Unit scale: 0.001 = mm→m, 0.01 = cm→m (default: {SCALE})",
    )
    parser.add_argument(
        "--tess", type=float, default=TESSELLATION,
        help=f"Tessellation tolerance in mm (default: {TESSELLATION})",
    )
    parser.add_argument(
        "--out", type=str, default="usd_shapes",
        help="Output directory (default: usd_shapes)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_summary(args.clearance, args.scale)

    def ex(shape, name):
        export_usd(shape, name, out_dir, scale=args.scale, tess_tol=args.tess)

    print("Generating shape pieces...")
    ex(make_cube(),             "cube")
    ex(make_cylinder(),         "cylinder")
    ex(make_triangular_prism(), "triangular_prism")
    ex(make_star(),             "star")
    ex(make_moon(),             "moon")

    print("\nGenerating shape sorter...")
    ex(make_sorter(args.clearance), "shape_sorter")

    print("\nGenerating shape sorter collision proxy...")
    ex(make_sorter_collision(args.clearance), "shape_sorter_collision")

    print(f"\nAll 7 files written to  ./{out_dir}/\n")


if __name__ == "__main__":
    main()