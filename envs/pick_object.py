"""pick_object — Vision-Tactile-Language-Action (VTLA) benchmark task.

Four objects are randomly placed on the table each episode:
  • oscar    — rigid statuette (oscar.usd)
  • mug      — rigid cylinder-like container (Can_d5cm.usd; swap for Mug.usd when available)
  • ball     — deformable soft sphere (Bar_Sphere.usd, UIPC/FEM)
  • nugget   — rigid rough-surfaced prism (RoughPrism.usd)

Each episode:
  1. A target object is selected at random.
  2. A natural-language instruction is drawn from that object's instruction pool and
     stored in self.instruction — it is saved to every observation frame under
     obs['language']['instruction'], producing a VTLA dataset.
  3. The expert grasps the target object (pre_move) and stands it upright on an
     orange pad placed in front of the robot (_play_once).

Asset notes
-----------
• Ball requires a UIPC-compatible sphere USD (Bar_Sphere.usd) with FEM material.
• Replace Can_d5cm.usd with a proper Mug.usd once that asset is available.
• RoughPrism.usd is used as the nugget — it has an irregular, bumpy surface.
"""

from ._base_task import *
import numpy as np

# ── Spawn region (robot-facing half of the table) ───────────────────────────
TABLE_SPAWN_CENTER_X = 0.52
TABLE_SPAWN_CENTER_Y = 0.0
TABLE_SPAWN_Z        = 0.001   # just above the table surface

GRID_SPACING_X = 0.13          # centre-to-centre spacing between grid rows
GRID_SPACING_Y = 0.13          # centre-to-centre spacing between grid columns

# Destination stand — orange pad in front of the robot
_STAND_BASE_POSE = Pose([0.35, 0.0, 0.005], [1, 0, 0, 0])

# ── Per-object configuration ─────────────────────────────────────────────────
OBJECT_CONFIGS: dict[str, dict] = {
    'oscar': {
        'asset_path': 'oscar.usd',
        'density': 500.0,
        'is_deformable': False,
        # grasp_z: height above object base where gripper fingers close
        'grasp_z': 0.04,
        # approach_extra_z: how far above grasp_z the pre-grasp waypoint sits
        'approach_extra_z': 0.08,
        'instructions': [
            "Pick up the oscar statuette and place it upright on the stand.",
            "Grasp the golden award figure and stand it upright on the platform.",
            "Lift the oscar trophy and position it upright on the stand.",
            "Take the award statuette and place it standing on the platform.",
            "Retrieve the oscar and set it upright on the orange pad.",
        ],
    },
    'mug': {
        # TODO: replace with Mug.usd once the asset is ready
        'asset_path': 'Can_d5cm.usd',
        'density': 300.0,
        'is_deformable': False,
        'grasp_z': 0.03,
        'approach_extra_z': 0.07,
        'instructions': [
            "Pick up the mug and place it upright on the stand.",
            "Grasp the ceramic mug and stand it on the platform.",
            "Lift the mug and position it upright on the stand.",
            "Take the cup and place it standing on the platform.",
            "Retrieve the mug and set it upright on the orange pad.",
        ],
    },
    'ball': {
        'asset_path': 'Bar_Sphere.usd',
        'density': 150.0,
        'is_deformable': True,   # uses UIPC FEM — requires FEM-capable USD
        'grasp_z': 0.025,
        'approach_extra_z': 0.07,
        'instructions': [
            "Pick up the soft ball and place it on the stand.",
            "Grasp the squishy ball and move it to the platform.",
            "Lift the deformable ball and set it on the stand.",
            "Take the soft sphere and place it on the platform.",
            "Retrieve the soft ball and move it onto the orange pad.",
        ],
    },
    'nugget': {
        'asset_path': 'RoughPrism.usd',
        'density': 800.0,
        'is_deformable': False,
        'grasp_z': 0.02,
        'approach_extra_z': 0.06,
        'instructions': [
            "Pick up the nugget and place it on the stand.",
            "Grasp the irregular nugget piece and set it on the platform.",
            "Lift the bumpy nugget and position it on the stand.",
            "Take the nugget and move it to the platform.",
            "Retrieve the rough nugget and place it on the orange pad.",
        ],
    },
}

OBJECT_NAMES: list[str] = list(OBJECT_CONFIGS.keys())   # fixed order for reproducibility


# ── Task config ──────────────────────────────────────────────────────────────

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim: int   = 800
    use_force_grasp: bool  = True
    grasp_force: float     = 8.0    # N — tune per sensor type / object


# ── Task implementation ──────────────────────────────────────────────────────

class Task(BaseTask):

    # ── Scene setup ──────────────────────────────────────────────────────────

    def create_actors(self):
        """Spawn all objects and the destination stand.

        Object positions are fixed here so the scene builds correctly;
        they are randomised each episode inside _reset_actors().
        """
        base_positions = self._grid_positions(len(OBJECT_NAMES))

        self._objects: dict[str, Actor] = {}
        for name, pos in zip(OBJECT_NAMES, base_positions):
            cfg = OBJECT_CONFIGS[name]
            spawn_pose = Pose(pos, [1, 0, 0, 0])

            if cfg['is_deformable']:
                actor = self._actor_manager.add_from_usd_file(
                    name=name,
                    asset_path=cfg['asset_path'],
                    pose=spawn_pose,
                    density=cfg['density'],
                )
            else:
                actor = self._actor_manager.add_rigid_from_usd_file(
                    name=name,
                    asset_path=cfg['asset_path'],
                    pose=spawn_pose,
                    density=cfg['density'],
                )
            self._objects[name] = actor

        # Destination stand — fixed, not randomised
        self._stand = self._actor_manager.add_rigid_from_usd_file(
            name='stand',
            asset_path='OrangePad.usd',
            pose=_STAND_BASE_POSE,
        )

    # ── Episode reset ─────────────────────────────────────────────────────────

    def _reset_actors(self):
        """Randomly assign grid slots to objects and pick a target."""
        # Select target object for this episode
        target_idx = int(self.rng.integers(0, len(OBJECT_NAMES)))
        self._target_name: str = OBJECT_NAMES[target_idx]

        # Randomly shuffle which grid slot each object occupies
        base_positions = self._grid_positions(len(OBJECT_NAMES))
        slot_order = self.rng.permutation(len(OBJECT_NAMES))

        for obj_idx, name in enumerate(OBJECT_NAMES):
            slot = int(slot_order[obj_idx])
            base_pos = base_positions[slot]
            # Small random noise so the robot cannot memorise exact positions
            noise = self.create_noise([0.015, 0.015, 0.0], [0, 0, np.pi / 12])
            pose = Pose(base_pos, [1, 0, 0, 0]).add_offset(noise)
            self._objects[name].set_pose(pose)

        # Set the episode instruction — this is picked up by _get_observations
        # and saved under obs['language']['instruction'] every step.
        instruction_pool = OBJECT_CONFIGS[self._target_name]['instructions']
        self.instruction = str(self.rng.choice(instruction_pool))

        self._success_steps: int = 0
        self.domain_rand_params = {
            'target_object': self._target_name,
            'instruction': self.instruction,
        }

    # ── Expert helpers ────────────────────────────────────────────────────────

    def _build_grasp_pose(self) -> tuple:
        """Return (grasp_pose, contact_idx) for the current target object."""
        target = self._objects[self._target_name]
        cfg    = OBJECT_CONFIGS[self._target_name]
        obj_p  = target.get_pose().p
        grasp_p = obj_p + np.array([0.0, 0.0, cfg['grasp_z']])
        grasp_pose = construct_grasp_pose(
            grasp_p,
            np.array([0.0, 0.0, 1.0]),   # approach from directly above
            np.array([1.0, 0.0, 0.0]),   # finger spread along x
        )
        contact_idx = target.register_point(pose=grasp_pose, type='contact')
        return grasp_pose, contact_idx

    # ── Expert policy ─────────────────────────────────────────────────────────

    def pre_move(self):
        """Approach and grasp the target object before data collection starts."""
        self.delay(10)
        _, contact_idx = self._build_grasp_pose()
        cfg = OBJECT_CONFIGS[self._target_name]

        self.move(self.atom.grasp_actor(
            self._objects[self._target_name],
            contact_point_id=contact_idx,
            pre_dis=cfg['approach_extra_z'],
            dis=0,
            is_close=False,
        ))
        self.move(self.atom.close_gripper(pos=0.0, force=self.cfg.grasp_force))

    def _play_once(self):
        """Lift the target object and place it upright on the stand."""
        # 1. Lift clear of the table and neighbouring objects
        self.move(self.atom.move_by_displacement(z=0.15), time_dilation_factor=0.5)

        # 2. Move laterally to above the stand
        stand_p = self._stand.get_pose().p
        ee_now  = self.atom.get_arm_pose()
        above_stand = Pose(
            [stand_p[0], stand_p[1], ee_now.p[2]],
            ee_now.q,
        )
        self.move(self.atom.move_to_pose(above_stand), time_dilation_factor=0.5)

        # 3. Lower onto stand surface
        self.move(self.atom.move_by_displacement(z=-0.10), time_dilation_factor=0.5)

        # 4. Release and retract
        self.move(self.atom.open_gripper(force=self.cfg.grasp_force, steps=30))
        self.move(self.atom.move_by_displacement(z=0.12), time_dilation_factor=0.5)

        # 5. Hold for final observation frames
        self.delay(20, is_save=True)

    # ── Success criterion ─────────────────────────────────────────────────────

    def check_success(self) -> bool:
        target    = self._objects[self._target_name]
        stand_p   = self._stand.get_pose().p
        obj_pos   = target.get_pose().p

        xy_ok = np.all(np.abs(obj_pos[:2] - stand_p[:2]) < 0.07)
        z_ok  = obj_pos[2] > stand_p[2] + 0.01

        in_zone = xy_ok and z_ok
        self._success_steps = (self._success_steps + 1) if in_zone else 0

        print(
            f'[pick_object] target={self._target_name}'
            f'  xy_err={np.linalg.norm(obj_pos[:2] - stand_p[:2]):.3f} m'
            f'  z_offset={obj_pos[2] - stand_p[2]:.3f} m'
            f'  success_steps={self._success_steps}'
        )
        # Require 5 consecutive frames on the stand to avoid false positives
        return self._success_steps >= 5

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _grid_positions(n: int) -> list[list[float]]:
        """Return n positions in a 2-column grid centred on the spawn area."""
        cols = 2
        rows = (n + cols - 1) // cols
        positions = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x = TABLE_SPAWN_CENTER_X + (row - (rows - 1) / 2.0) * GRID_SPACING_X
            y = TABLE_SPAWN_CENTER_Y + (col - (cols - 1) / 2.0) * GRID_SPACING_Y
            positions.append([x, y, TABLE_SPAWN_Z])
        return positions
