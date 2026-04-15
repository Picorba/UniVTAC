#!/usr/bin/env python3
"""
Generate shape sorter 3D assets for robotic manipulation simulation.

Produces 6 STL files:
  cube.stl, cylinder.stl, triangular_prism.stl, star.stl, moon.stl
  shape_sorter.stl  — single solid block, 5 blind holes + clearance

Install:
  pip install cadquery

Usage:
  python generate_shape_sorter.py
  python generate_shape_sorter.py --clearance 1.0 --out ./my_output
"""

import argparse
import math
from pathlib import Path

import cadquery as cq


# ══════════════════════════════════════════════════════════════════════════════
# Default parameters  (all dimensions in mm)
# ══════════════════════════════════════════════════════════════════════════════

CLEARANCE    = 2   # uniform gap between shape piece and sorter hole wall

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
    Return 2n vertices of an n-pointed star.
    First tip points upward (+Y).  Winding is counter-clockwise.
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

    When clearance > 0, each edge is moved outward by exactly `clearance` mm.
    Correct formula for uniform edge offset on an equilateral triangle:
        new_side = side + 2 * sqrt(3) * clearance
    Winding: counter-clockwise  (top → bottom-right → bottom-left).
    """
    s = side + 2.0 * math.sqrt(3) * clearance
    h = s * math.sqrt(3) / 2.0
    return [(0.0, 2 * h / 3), (s / 2, -h / 3), (-s / 2, -h / 3)]


# ══════════════════════════════════════════════════════════════════════════════
# Shape pieces — all centred at world origin
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
        .translate((0, 0, -h / 2))      # re-centre in Z
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
    Both solids are extruded by h; subtraction happens in 3D, then centred in Z.
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

    Clearance strategy per shape
    ────────────────────────────
    cube      : each face expanded by clearance
    cylinder  : radius expanded by clearance/2  (uniform radial gap)
    triangle  : each edge shifted outward using the exact equilateral formula
    star      : outer tips expanded; inner valleys slightly widened
    moon      : outer circle expanded; inner cut-circle contracted
    """
    c = clearance

    # Hole centre X positions (sorter box centred at X=0)
    xs = [-SORTER_W / 2 + MARGIN + i * HOLE_SPACING for i in range(5)]

    # ── Z reference values ────────────────────────────────────────────────────
    # Sorter box is Z-centred: Z ∈ [−SORTER_H/2, +SORTER_H/2]
    # All holes are cut from the top face (Z = +SORTER_H/2) downward.
    #
    # cq.box / cq.cylinder are centred solids:
    #   place their centre at z_cen so their top edge = +SORTER_H/2
    z_cen = SORTER_H / 2 - HOLE_DEPTH / 2
    #
    # polyline.extrude(d) produces a solid from Z=0 to Z=d:
    #   translate its base to z_bot so its top edge = +SORTER_H/2
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
    # Expand outer tips; slightly widen inner valleys for clearance at concavities
    pts = star_pts(STAR_RO + c / 2, STAR_RI - c * 0.25)
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
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def export(shape: cq.Workplane, name: str, out_dir: Path):
    path = str(out_dir / f"{name}.stl")
    cq.exporters.export(shape, path)
    print(f"  ✓  {name}.stl")


def print_summary(clearance: float):
    half_w = SORTER_W / 2
    xs = [-half_w + MARGIN + i * HOLE_SPACING for i in range(5)]
    labels = ["cube", "cylinder", "triangle", "star", "moon"]
    print(f"\n  Sorter block   : {SORTER_W} × {SORTER_D} × {SORTER_H} mm")
    print(f"  Hole depth     : {HOLE_DEPTH} mm   floor: {SORTER_H - HOLE_DEPTH} mm")
    print(f"  Clearance      : {clearance} mm")
    print(f"  Hole layout (X): {[round(x, 1) for x in xs]}")
    for label, x in zip(labels, xs):
        wall_left  = round(half_w - abs(x) - _half_footprint(label, clearance), 1)
        print(f"    {label:<14} @ X = {x:+6.1f}   min wall ≈ {wall_left} mm")
    print()


def _half_footprint(shape: str, c: float) -> float:
    """Approximate max half-width of each hole for wall-thickness info."""
    return {
        "cube":     (CUBE_S + c) / 2,
        "cylinder": CYL_R + c / 2,
        "triangle": (TRI_S + 2 * math.sqrt(3) * c / 2) / 2 * 1.15,
        "star":     STAR_RO + c / 2,
        "moon":     MOON_RO + c / 2,
    }[shape]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate shape sorter STL assets for robotic manipulation"
    )
    parser.add_argument(
        "--clearance", type=float, default=CLEARANCE,
        help=f"Hole clearance in mm (default: {CLEARANCE})",
    )
    parser.add_argument(
        "--out", type=str, default="stl_shapes",
        help="Output directory (default: stl_shapes)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = args.clearance

    print_summary(c)

    print("Generating shape pieces...")
    export(make_cube(),             "cube",             out_dir)
    export(make_cylinder(),         "cylinder",         out_dir)
    export(make_triangular_prism(), "triangular_prism", out_dir)
    export(make_star(),             "star",             out_dir)
    export(make_moon(),             "moon",             out_dir)

    print("\nGenerating shape sorter...")
    export(make_sorter(c), "shape_sorter", out_dir)

    print(f"\nAll 6 files written to  ./{out_dir}/")
    print(
        "Next step:\n"
        f"  blender --background --python process_stl.py -- "
        f"--input ./{out_dir} --output ./usd_output --scale 0.01\n"
    )


if __name__ == "__main__":
    main()