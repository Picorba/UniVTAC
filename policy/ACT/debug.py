import h5py
import sys

path = sys.argv[1]

def print_structure(name, obj):
    print(name, "->", type(obj).__name__, end="")
    if isinstance(obj, h5py.Dataset):
        print(f"  shape={obj.shape} dtype={obj.dtype}", end="")
    print()

with h5py.File(path, "r") as f:
    print(f"=== Structure of {path} ===")
    f.visititems(print_structure)