"""Generate Hammer OBJ mesh files for the find_equilibrium task.

Three variants are produced with different head sizes, yielding different CoM positions
along the handle (head size is the proxy for head-to-handle mass ratio):

  Variant     head (depth × width × height)   CoM from handle centre
  --------    ----------------------------    ----------------------
  light       0.03 × 0.05 × 0.03 m           ~0.068 m
  medium      0.04 × 0.07 × 0.04 m           ~0.106 m
  heavy       0.05 × 0.09 × 0.05 m           ~0.136 m

All share the same handle: 0.30 m long, 0.015 × 0.015 m cross-section.
The head is attached at the +x end of the handle.

Run standalone (no Isaac Sim required):
    python scripts/gen_hammer.py

Then convert each OBJ to USD inside the Docker / Isaac Sim environment:
    for v in light medium heavy; do
        python scripts/convert.py \\
            --input  assets/objects/Hammer_${v}.obj \\
            --output assets/objects/Hammer_${v}.usd
    done
"""

import numpy as np
from pathlib import Path

# ── Shared handle geometry ────────────────────────────────────────────────────
HANDLE_L = 0.30    # length along x-axis (m)
HANDLE_W = 0.015   # square cross-section side (m)

# ── Head dimensions per variant ───────────────────────────────────────────────
VARIANTS: dict[str, dict] = {
    'light':  dict(depth=0.03, width=0.05, height=0.03),
    'medium': dict(depth=0.04, width=0.07, height=0.04),
    'heavy':  dict(depth=0.05, width=0.09, height=0.05),
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def compute_com_x(depth: float, width: float, height: float) -> float:
    """CoM position along local x-axis (from handle centre) for uniform density."""
    v_handle = HANDLE_L * HANDLE_W ** 2
    v_head   = depth * width * height
    head_cx  = HANDLE_L / 2 + depth / 2   # head centre along x
    return v_head * head_cx / (v_handle + v_head)


def box_vertices(cx: float, cy: float, cz: float,
                 hx: float, hy: float, hz: float) -> np.ndarray:
    """Return (8, 3) vertex array for a box with given centre and half-extents."""
    return np.array([
        [cx-hx, cy-hy, cz-hz],  # 0
        [cx+hx, cy-hy, cz-hz],  # 1
        [cx+hx, cy+hy, cz-hz],  # 2
        [cx-hx, cy+hy, cz-hz],  # 3
        [cx-hx, cy-hy, cz+hz],  # 4
        [cx+hx, cy-hy, cz+hz],  # 5
        [cx+hx, cy+hy, cz+hz],  # 6
        [cx-hx, cy+hy, cz+hz],  # 7
    ])


def box_faces(offset: int = 0) -> np.ndarray:
    """Return (12, 3) triangle index array for a box (outward normals).

    Face normals verified with right-hand rule:
      bottom(-z) [0,3,1],[1,3,2]  top(+z) [4,5,7],[5,6,7]
      front(-y)  [0,1,5],[0,5,4]  back(+y)[2,7,6],[2,3,7]
      left(-x)   [0,4,7],[0,7,3]  right(+x)[1,2,6],[1,6,5]
    """
    f = np.array([
        [0, 3, 1], [1, 3, 2],   # -z
        [4, 5, 7], [5, 6, 7],   # +z
        [0, 1, 5], [0, 5, 4],   # -y
        [2, 7, 6], [2, 3, 7],   # +y
        [0, 4, 7], [0, 7, 3],   # -x
        [1, 2, 6], [1, 6, 5],   # +x
    ], dtype=int)
    return f + offset


# ── OBJ writer ────────────────────────────────────────────────────────────────

def write_hammer_obj(path: Path, depth: float, width: float, height: float) -> tuple[float, float]:
    """Write a hammer OBJ file; return (com_x, spawn_z)."""
    hL = HANDLE_L / 2
    hW = HANDLE_W / 2

    # Handle: centred at origin
    v_handle = box_vertices(0.0, 0.0, 0.0, hL, hW, hW)
    f_handle = box_faces(offset=0)

    # Head: attached at the +x end of the handle
    head_cx = hL + depth / 2
    v_head = box_vertices(head_cx, 0.0, 0.0, depth/2, width/2, height/2)
    f_head = box_faces(offset=8)

    verts = np.vstack([v_handle, v_head])   # 16 vertices
    faces = np.vstack([f_handle, f_head])   # 24 triangles

    com_x   = compute_com_x(depth, width, height)
    spawn_z = height / 2 + 0.002   # head half-height + 2 mm clearance above table

    with open(path, 'w') as f:
        f.write(f"# Hammer ({path.stem}): head {depth:.3f}×{width:.3f}×{height:.3f} m\n")
        f.write(f"# CoM along handle:  {com_x:.4f} m from handle centre\n")
        f.write(f"# Spawn z (flat):    {spawn_z:.4f} m (bottom of head at z=0)\n\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for tri in faces:
            # OBJ indices are 1-based
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    return com_x, spawn_z


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = Path('assets/objects')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating hammer variants:\n")
    print(f"  {'Variant':<10}  {'CoM (m)':<10}  {'spawn_z (m)':<12}  Output")
    print("  " + "-" * 60)
    for name, params in VARIANTS.items():
        path = out_dir / f"Hammer_{name}.obj"
        com_x, spawn_z = write_hammer_obj(path, **params)
        print(f"  {name:<10}  {com_x:<10.4f}  {spawn_z:<12.4f}  {path}")

    print("\nConvert to USD (run inside Docker / Isaac Sim env):\n")
    for name in VARIANTS:
        print(f"  python scripts/convert.py "
              f"--input assets/objects/Hammer_{name}.obj "
              f"--output assets/objects/Hammer_{name}.usd")
