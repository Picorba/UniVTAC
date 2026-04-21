import bpy
import os
from pathlib import Path

def process_stl_to_usd(input_dir: str, output_dir: str, scale: float = 0.01):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_files = list(input_dir.glob("*.stl")) + list(input_dir.glob("*.STL"))
    print(f"Found {len(stl_files)} STL files")

    for stl_path in stl_files:
        print(f"Processing: {stl_path.name}")

        # ── 1. Clean scene ───────────────────────────────────────────
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # ── 2. Import STL ────────────────────────────────────────────
        bpy.ops.wm.stl_import(filepath=str(stl_path))
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

        # ── 3. Recenter — origin to geometry, then move to world 0,0,0
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0.0, 0.0, 0.0)

        # ── 4. Apply scale x0.01 ─────────────────────────────────────
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=False)

        # ── 5. Export USD ────────────────────────────────────────────
        out_path = output_dir / (stl_path.stem + ".usd")
        bpy.ops.wm.usd_export(
            filepath=str(out_path),
            export_meshes=True,
            export_normals=True,
            export_uvmaps=True,
            export_materials=False,   # STLs have no materials
            root_prim_path="/World",
            export_global_forward_selection='NEGATIVE_Z',  # match Isaac Sim convention
            export_global_up_selection='Y',
        )
        print(f"  → {out_path}")

    print("Done.")

if __name__ == "__main__":
    process_stl_to_usd(
        input_dir="/home/pierre/Downloads/stl_files",
        output_dir="./usd_output",
        scale=0.01,
    )