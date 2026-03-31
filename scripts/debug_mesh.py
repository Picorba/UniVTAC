from pxr import Usd, UsdGeom
import argparse

parser = argparse.ArgumentParser(description="Remove embedded tet mesh attributes from USD render mesh")
parser.add_argument("path", help="Path to the USD file")
parser.add_argument("--output", help="Output path (default: overwrite input)", default=None)
args = parser.parse_args()

stage = Usd.Stage.Open(args.path)

TET_ATTRIBUTES = [
    "tet_indices",
    "tet_points",
    "tet_surf_indices",
    "tet_surf_points",
]

for prim in stage.Traverse():
    if prim.GetTypeName() == "Mesh":
        print(f"Processing mesh: {prim.GetPath()}")
        for attr_name in TET_ATTRIBUTES:
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                prim.RemoveProperty(attr_name)
                print(f"  Removed: {attr_name}")
            else:
                print(f"  Not found (skipped): {attr_name}")

output_path = args.output or args.path
stage.GetRootLayer().Export(output_path)
print(f"\nSaved to: {output_path}")