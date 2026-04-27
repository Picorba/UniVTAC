# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a OBJ/STL/FBX/GLB/USD into USD format (with tetrahedral mesh generation).

Supported input formats:
  - OBJ, STL, FBX, GLB  → converted via omni.kit.asset_converter, then tet mesh is added
  - USD, USDA, USDC     → conversion step is skipped; tet mesh is added directly

positional arguments:
  --input     The path to the input mesh file or directory.
  --output    The path to store the USD file or output directory.

optional arguments:
  -h, --help                    Show this help message and exit
  --make-instanceable           Make the asset instanceable for efficient cloning. (default: False)
  --collision-approximation     The method used for approximating collision mesh. (default: convexDecomposition)
  --mass                        The mass (in kg) to assign to the converted asset. (default: None)
  --show                        Show trimesh visualization of the generated tetrahedral mesh.
  --edge-length-r               Relative edge length for tet mesh (smaller = finer). (default: 0.25)
  --backend                     Backend to use for tet mesh gen: ftetwild or tetgen. (default: ftetwild)
"""

"""Launch Isaac Sim Simulator first."""

import os
import asyncio
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file (or USD) into USD format with tet mesh.")
parser.add_argument("--input", "-i", type=str, help="The path to the input mesh file.", default='assets/objects/ipt')
parser.add_argument("--output", "-o", type=str, help="The path to store the USD file.", default='assets/objects/opt')
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="none",
    choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere", "meshSimplification", "none"],
    help=(
        'The method used for approximating collision mesh. Defaults to "none" '
        "because UIPC handles its own contact — do not add a PhysX collision mesh "
        "for UIPC FEM objects. Set to convexDecomposition etc. only for PhysX rigid bodies."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
parser.add_argument(
    '--show', action='store_true',
    help='Show trimesh visualization of the generated tetrahedral mesh.'
)
parser.add_argument(
    '--edge-length-r',
    type=float,
    default=0.25,
    help='Relative edge length for tet mesh (smaller = finer). Default 0.25 matches UI converter.'
)
parser.add_argument(
    '--backend',
    type=str,
    choices=["ftetwild", "tetgen"],
    default="ftetwild",
    help="Backend to use for the tet mesh gen"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # enforce headless mode

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np

import omni.kit.app

from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path

import omni
import omni.kit.commands
from omni.physx.scripts import deformableUtils
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import export_prim_to_file

from pathlib import Path
from pxr import Usd, Sdf, Gf
# After — matches tacex_uipc.utils used in UipcObject
from tacex_uipc.utils import MeshGenerator, TetMeshCfg
# Formats that require the omni.kit.asset_converter step
MESH_FORMATS = {".obj", ".stl", ".fbx", ".glb"}
# Formats that are already USD — skip conversion
USD_FORMATS = {".usd", ".usda", ".usdc"}
ALL_SUPPORTED = MESH_FORMATS | USD_FORMATS


def visualize_tet(tet_points, tet_indices, is_save=False):
    import trimesh
    pts = tet_points
    faces = []
    for i in range(0, len(tet_indices), 4):
        v0, v1, v2, v3 = tet_indices[i:i+4]
        faces.extend([
            [v0, v2, v1],
            [v1, v2, v3],
            [v0, v1, v3],
            [v0, v3, v2]
        ])
    msh = trimesh.Trimesh(vertices=pts, faces=faces)
    trimesh.Scene([msh]).show()

    if is_save:
        msh.export('tet_mesh_visualize.glb')


class MeshConverter(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX / GLB / USD format to a USD file with tet mesh data.

    When the input is already a USD file, the omni.kit.asset_converter step is skipped entirely and
    the tet mesh attributes are added directly to the existing mesh prims.
    """

    cfg: MeshConverterCfg

    def __init__(self, cfg: MeshConverterCfg):
        super().__init__(cfg=cfg)

    @staticmethod
    def set_attr(prim: Usd.Prim, attr_name: str, attr_type, attr_value):
        prim.CreateAttribute(attr_name, attr_type).Set(attr_value)

    # ------------------------------------------------------------------
    # Main entry point (called by AssetConverterBase)
    # ------------------------------------------------------------------

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate or enrich a USD file with tet mesh data.

        If the input is already a USD/USDA/USDC file the conversion step is
        skipped and tet attributes are written directly into a copy of the
        source stage.  Otherwise the standard OBJ/STL/FBX/GLB → USD path is
        followed before tet generation.
        """
        input_ext = os.path.splitext(cfg.asset_path)[1].lower()

        if input_ext in USD_FORMATS:
            self._process_usd_input(cfg)
        else:
            self._process_mesh_input(cfg)

    # ------------------------------------------------------------------
    # Path A: input is already a USD file
    # ------------------------------------------------------------------

    def _process_usd_input(self, cfg: MeshConverterCfg):
        """Add tet mesh attributes to an existing USD stage (no format conversion)."""
        import shutil

        src_path = cfg.asset_path
        dst_path = self.usd_path  # set by AssetConverterBase from cfg.usd_dir / cfg.usd_file_name

        # Copy source USD to the output location (so we never mutate the original)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy2(src_path, dst_path)

        # Open the copied stage
        stage = Usd.Stage.Open(dst_path)
        stage.Reload()
        stage_id = UsdUtils.StageCache.Get().Insert(stage)

        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            # Fall back: use the first root prim
            root_prims = list(stage.GetPseudoRoot().GetChildren())
            if not root_prims:
                raise RuntimeError(f"USD file '{src_path}' has no prims.")
            default_prim = root_prims[0]
            stage.SetDefaultPrim(default_prim)

        # Walk all prims and process every Mesh we find
        self._enrich_mesh_prims(stage, default_prim, cfg)

        # Apply physics schemas to the default (root) prim if requested
        if cfg.mass_props is not None:
            schemas.define_mass_properties(
                prim_path=default_prim.GetPath(), cfg=cfg.mass_props, stage=stage
            )
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(
                prim_path=default_prim.GetPath(), cfg=cfg.rigid_props, stage=stage
            )

        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)

    def _enrich_mesh_prims(self, stage: Usd.Stage, root_prim: Usd.Prim, cfg: MeshConverterCfg):
        """Recursively find Mesh prims under *root_prim* and add tet attributes + collision."""
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() != "Mesh":
                continue

            # Collision
            if cfg.collision_props is not None:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                schemas.define_collision_properties(
                    prim_path=prim.GetPath(), cfg=cfg.collision_props, stage=stage
                )

            # Tet mesh generation
            usd_mesh = UsdGeom.Mesh(prim)
            tet_points, tet_indices, surf_points, tet_surf_indices = self.gen_tet(
                usd_mesh, backend=args_cli.backend
            )
            print(f"  Mesh '{prim.GetPath()}' → tet points: {len(tet_points)}, tets: {len(tet_indices) // 4}")

            self.set_attr(prim, 'tet_points',      Sdf.ValueTypeNames.Float3Array, tet_points)
            self.set_attr(prim, 'tet_indices',      Sdf.ValueTypeNames.UIntArray,   tet_indices)
            self.set_attr(prim, 'tet_surf_points',  Sdf.ValueTypeNames.Float3Array, surf_points)
            self.set_attr(prim, 'tet_surf_indices', Sdf.ValueTypeNames.UIntArray,   tet_surf_indices)

            if args_cli.show:
                visualize_tet(tet_points, tet_indices)

    # ------------------------------------------------------------------
    # Path B: input is OBJ / STL / FBX / GLB  (original logic, unchanged)
    # ------------------------------------------------------------------

    def _process_mesh_input(self, cfg: MeshConverterCfg):
        """Original conversion flow for non-USD mesh formats."""
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()

        if not Tf.IsValidIdentifier(mesh_file_basename):
            mesh_file_basename_original = mesh_file_basename
            mesh_file_basename = Tf.MakeValidIdentifier(mesh_file_basename)
            omni.log.warn(
                f"Input file name '{mesh_file_basename_original}' is an invalid identifier for the mesh prim path."
                f" Renaming it to '{mesh_file_basename}' for the conversion."
            )

        # --- Step 1: omni.kit.asset_converter ---
        # Convert to an intermediate file so the wrapper stage (Step 2) can
        # reference it without overwriting the file it references.
        raw_usd_path = os.path.join(
            self.usd_dir,
            os.path.splitext(self.usd_file_name)[0] + "_meshes.usd",
        )
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(in_file=cfg.asset_path, out_file=raw_usd_path)
        )

        # --- Step 2: rebuild stage structure ---
        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)
        base_prim = temp_stage.DefinePrim(f"/{mesh_file_basename}", "Xform")
        base_prim.GetReferences().AddReference(raw_usd_path)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(self.usd_path)

        stage = Usd.Stage.Open(self.usd_path)
        stage.Reload()
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}")

        # --- Step 3: collision + tet mesh on each Mesh child ---
        for child_mesh_prim in geom_prim.GetChildren():
            if child_mesh_prim.GetTypeName() == "Mesh":
                if cfg.collision_props is not None:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_mesh_prim)
                    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                    schemas.define_collision_properties(
                        prim_path=child_mesh_prim.GetPath(), cfg=cfg.collision_props, stage=stage
                    )

        stage.SetDefaultPrim(xform_prim)
        omni.kit.commands.execute(
            "CreateDefaultXformOnPrimCommand",
            prim_path=xform_prim.GetPath(),
            **{"stage": stage},
        )

        geom_xform = UsdGeom.Xform(geom_prim)
        geom_xform.ClearXformOpOrder()

        rotate_attr = geom_prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr:
            geom_prim.RemoveProperty(rotate_attr.GetName())

        translate_op = geom_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*cfg.translation))
        orient_op = geom_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(*cfg.rotation))
        scale_op = geom_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(*cfg.scale))

        for child_mesh_prim in geom_prim.GetChildren():
            if child_mesh_prim.GetTypeName() != "Mesh":
                continue
            usd_mesh = UsdGeom.Mesh(child_mesh_prim)
            tet_points, tet_indices, surf_points, tet_surf_indices = self.gen_tet(
                usd_mesh, backend=args_cli.backend
            )
            print(f"  total tet points: {len(tet_points)}, total tets: {len(tet_indices) // 4}")
            self.set_attr(child_mesh_prim, 'tet_points',      Sdf.ValueTypeNames.Float3Array, tet_points)
            self.set_attr(child_mesh_prim, 'tet_indices',      Sdf.ValueTypeNames.UIntArray,   tet_indices)
            self.set_attr(child_mesh_prim, 'tet_surf_points',  Sdf.ValueTypeNames.Float3Array, surf_points)
            self.set_attr(child_mesh_prim, 'tet_surf_indices', Sdf.ValueTypeNames.UIntArray,   tet_surf_indices)

            if args_cli.show:
                visualize_tet(tet_points, tet_indices)

        # --- Step 4: instanceable handling ---
        if cfg.make_instanceable:
            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )
            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)
            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(
                self.usd_instanceable_meshes_path, primPath=geom_prim_path
            )
            geom_undef_prim.SetInstanceable(True)

        # --- Step 5: physics schemas ---
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)

        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)

    # ------------------------------------------------------------------
    # Tet mesh generation (unchanged)
    # ------------------------------------------------------------------

    def gen_tet(self, prim: UsdGeom.Mesh, backend='ftetwild'):
        """Generate a tetrahedral mesh from a USD mesh prim.

        Parameters match UipcObject's runtime fallback exactly so that
        precomputed tet attributes are bit-for-bit consistent with what
        the simulator would generate on the fly.
        """
        if backend == 'tetgen':
            import tetgen
            import pymeshfix
            import pyvista as pv

            points = np.array(prim.GetPointsAttr().Get())
            triangles = np.array(deformableUtils.triangulate_mesh(prim))

            import trimesh
            msh = trimesh.Trimesh(vertices=points, faces=triangles.reshape(-1, 3))
            msh.merge_vertices(digits_vertex=8)
            msh.update_faces(msh.unique_faces())
            msh.update_faces(msh.nondegenerate_faces())

            v_clean, f_clean = pymeshfix.clean_from_arrays(
                msh.vertices,
                msh.faces.astype(np.int32),
                joincomp=False,
                remove_smallest_components=False
            )

            tg = tetgen.TetGen(v_clean, f_clean)
            tg.tetrahedralize()

            grid = tg.grid
            tet_points = grid.points
            cells = grid.cells_dict[10]
            tet_indices = cells.flatten().tolist()

            surface_polydata: pv.PolyData = grid.extract_surface(
                pass_pointid=False,
                pass_cellid=False,
                nonlinear_subdivision=1,
                progress_bar=False
            )
            faces = surface_polydata.faces
            surf_faces = faces.reshape(-1, 4)[:, 1:4]
            raw_surf_points = np.array(surface_polydata.points)
            raw_surf_indices = surf_faces.astype(np.int32)

            surf_points_clean, surf_faces_clean = pymeshfix.clean_from_arrays(
                raw_surf_points,
                raw_surf_indices,
                joincomp=False,
                remove_smallest_components=False
            )
            return tet_points, cells.flatten().tolist(), surf_points_clean.tolist(), surf_faces_clean.flatten().tolist()

        else:  # ftetwild — mirrors UipcObject.__init__ fallback exactly
            mesh_gen = MeshGenerator(config=TetMeshCfg(
                stop_quality=8,    # was 6 in converter, 8 in UipcObject  ← aligned
                max_its=100,
                edge_length_r=1 / 5,  # was args_cli.edge_length_r (default 0.25), UipcObject uses 0.2 ← aligned
                epsilon_r=0.001    # was 0.01 in converter, 0.001 in UipcObject ← aligned
            ))
            return mesh_gen.generate_tet_mesh_for_prim(prim)

    # ------------------------------------------------------------------
    # Static helper: omni.kit.asset_converter wrapper (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    async def _convert_mesh_to_usd(in_file: str, out_file: str, load_materials: bool = True) -> bool:
        enable_extension("omni.kit.asset_converter")

        import omni.kit.asset_converter

        converter_context = omni.kit.asset_converter.AssetConverterContext()
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        converter_context.merge_all_meshes = True
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        converter_context.use_double_precision_to_usd_transform_op = True

        instance = omni.kit.asset_converter.get_instance()
        task = instance.create_converter_task(in_file, out_file, None, converter_context)
        success = await task.wait_until_finished()
        if not success:
            raise RuntimeError(f"Failed to convert {in_file} to USD. Error: {task.get_error_message()}")
        return success


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def convert_mesh(
    input_path: Path,
    output_path: Path,
    make_instanceable: bool = False,
    collision_approximation: str = "convexDecomposition",
    mass: float = None,
):
    """Convert or enrich a mesh / USD file and return the output USD path."""
    input_path = input_path.absolute()
    output_path = output_path.absolute()

    if not check_file_path(str(input_path)):
        raise ValueError(f"Invalid mesh file path: {input_path}")

    if mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None

    if collision_approximation == "none":
        collision_props = None
    else:
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=str(input_path),
        force_usd_conversion=True,
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        make_instanceable=make_instanceable,
        collision_approximation=collision_approximation,
    )
    mesh_converter = MeshConverter(mesh_converter_cfg)
    return Path(mesh_converter.usd_path)


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    global args_cli
    input_path = Path(args_cli.input)
    output_path = Path(args_cli.output)

    process_list = []
    if input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            output_path = output_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        for file in input_path.iterdir():
            if file.suffix.lower() in ALL_SUPPORTED:
                out_file = output_path / (file.stem + ".usd")
                process_list.append((file, out_file))
    else:
        if output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / (input_path.stem + ".usd")
        else:
            out_file = output_path
        process_list.append((input_path, out_file))

    if not process_list:
        print(f"No supported files found. Supported formats: {sorted(ALL_SUPPORTED)}")
        return

    total_files = len(process_list)
    print(f"{total_files} file(s) to process:")
    for idx, (i, o) in enumerate(process_list):
        ext = i.suffix.lower()
        mode = "enriching USD" if ext in USD_FORMATS else "converting mesh"
        print(f"[{idx + 1}/{total_files}] {mode}: {i}  →  {o}")
        usd_path = convert_mesh(
            i, o,
            make_instanceable=args_cli.make_instanceable,
            collision_approximation=args_cli.collision_approximation,
            mass=args_cli.mass,
        )
        print(f"[{idx + 1}/{total_files}] Done. Output saved at: {usd_path}")


def visualize(name):
    usd_path = Path(f'assets/objects/{name}.usd')
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(f'/{name}/mesh')
    tet_points = prim.GetAttribute('tet_points').Get()
    tet_indices = prim.GetAttribute('tet_indices').Get()
    visualize_tet(tet_points, tet_indices, is_save=True)


if __name__ == "__main__":
    main()
    simulation_app.close()