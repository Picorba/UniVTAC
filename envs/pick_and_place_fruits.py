"""puts_fruits_bowl — Vision-Tactile-Language-Action (VTLA) task.

2–4 fruits are randomly selected and spawned on the table each episode.
A natural-language instruction specifies which fruit the robot must grasp
and drop into a bowl placed in front of the robot.

Episode flow
------------
2. A target fruit is chosen; a language instruction is sampled from its pool.
3. pre_move: open gripper.
4. _play_once: grasp the target fruit and drop it into the bowl.
5. check_success: target fruit is inside the bowl zone.
"""

from ._base_task import *
import numpy as np
from math import sqrt
from tacex_uipc import (
    UipcObject,
    UipcObjectCfg,
)

# ── Offscreen parking ─────────────────────────────────────────────────────────
OFFSCREEN_X      = 3.0
OFFSCREEN_Y_START = 3.0
OFFSCREEN_Z      = 3.0
GRID_SPACING_Y   = 1

# ── Spawn heights ─────────────────────────────────────────────────────────────
SPAWN_SAFE_Z   = 1.0    # intermediate waypoint height during reset
TABLE_SPAWN_Z  = 0.1   # resting height on the table surface

# ── Success thresholds ────────────────────────────────────────────────────────
_SUCCESS_XY_TOL = 0.12   # m — horizontal tolerance to be "in the bowl"
_SUCCESS_Z_MIN  = 0.08   # m — fruit must be below bowl_z + this value
# claude --resume bfbc07f7-77ae-4de4-8122-d3b5b4de17b6 FOR FRICTION PER OBJECT
# ── Fruit asset configurations ────────────────────────────────────────────────
FRUITS_CONFIGS: dict[str, dict] = {
    'pear': {
        'asset_path': 'fruits_tet/pear.usd',
        'density': 1010,
        'm_kappa' : 200,
        'grasp_z': -0.01,
        'depth_threshold': 26,

        'instructions': [
            "Grasp the pear and put it in the bowl.",
            "Pick up the pear and drop it into the bowl.",
            "Take the red pear and place it in the bowl.",
            "Grab the pear and put it inside the bowl.",
            "Pick the pear and drop it into the fruit bowl.",
        ],
    },
    'banana': {
        'asset_path': 'fruits_tet/banana.usd',
        'density': 950,
        'grasp_z': -0.015,
        'm_kappa': 50,
        'depth_threshold': 25,

        'instructions': [
            "Grasp the banana and put it in the bowl.",
            "Pick up the banana and drop it into the bowl.",
            "Take the yellow banana and place it in the bowl.",
            "Grab the banana and put it inside the bowl.",
            "Pick the banana and drop it into the fruit bowl.",
        ],
    },
    'cherry': {
        'asset_path': 'fruits_tet/cherry.usd',
        'density': 1100,
        'grasp_z': -0.01,
        'm_kappa': 1500,
        'depth_threshold': 27,
        'instructions': [
            "Grasp the cherry and put it in the bowl.",
            "Pick up the cherry and drop it into the bowl.",
            "Take the small cherry and place it in the bowl.",
            "Grab the cherry and put it inside the bowl.",
            "Pick the cherry and drop it into the fruit bowl.",
        ],
    },
    'lemon': {
        'asset_path': 'fruits_tet/lemon.usd',
        'density': 1050,
        'grasp_z': -0.015,
        'm_kappa': 600,
        'depth_threshold': 26,
        'instructions': [
            "Grasp the lemon and put it in the bowl.",
            "Pick up the lemon and drop it into the bowl.",
            "Take the lemon and place it in the bowl.",
            "Grab the lemon and put it inside the bowl.",
            "Pick the lemon cluster and drop it into the fruit bowl.",
        ],
    },
    'orange': {
        'asset_path': 'fruits_tet/orange.usd',
        'density': 860,
        'grasp_z': -0.015,
        'm_kappa': 400,
        'depth_threshold': 26,
        'instructions': [
            "Grasp the orange and put it in the bowl.",
            "Pick up the orange and drop it into the bowl.",
            "Take the round orange and place it in the bowl.",
            "Grab the orange and put it inside the bowl.",
            "Pick the orange and drop it into the fruit bowl.",
        ],
    },}
"""'pear': {
    'asset_path': 'fruits_tet/banana.usd',
    'density': 10.0,
    'grasp_z': 0.010,
    'instructions': [
        "Grasp the pear and put it in the bowl.",
        "Pick up the pear and drop it into the bowl.",
        "Take the pear-shaped fruit and place it in the bowl.",
        "Grab the pear and put it inside the bowl.",
        "Pick the pear and drop it into the fruit bowl.",
    ],
},"""
"""'pineapple': {
    'asset_path': 'fruits_tet/banana.usd',
    'density': 10.0,
    'grasp_z': 0.015,
    'instructions': [
        "Grasp the pineapple and put it in the bowl.",
        "Pick up the pineapple and drop it into the bowl.",
        "Take the pineapple and place it in the bowl.",
        "Grab the pineapple and put it inside the bowl.",
        "Pick the pineapple and drop it into the fruit bowl.",
    ],"""
"""'coconut': {
    'asset_path': 'fruits_tet/banana.usd',
    'density': 10.0,
    'grasp_z': 0.012,
    'instructions': [
        "Grasp the coconut and put it in the bowl.",
        "Pick up the coconut and drop it into the bowl.",
        "Take the round coconut and place it in the bowl.",
        "Grab the coconut and put it inside the bowl.",
        "Pick the coconut and drop it into the fruit bowl.",
    ],"""

"""fruit = 'pear'
FRUITS_CONFIGS = {fruit: FRUITS_CONFIGS[fruit]}"""

ALL_FRUIT_NAMES: list[str] = list(FRUITS_CONFIGS.keys())

# ── Bowl configuration ────────────────────────────────────────────────────────
# ── Bowl configuration ────────────────────────────────────────────────────────
BOWL_ASSET_PATH = "fruits_tet/bowl.usd"
BOWL_BASE_POSE  = Pose([0.5, 0.0, 0.041], [0.7071, 0, 0, 0.7071])

# ── Bowl spawn bounds (centred around table middle) ───────────────────────────
BOWL_X_MIN, BOWL_X_MAX = 0.35, 0.65
BOWL_Y_MIN, BOWL_Y_MAX = -0.20, 0.20
BOWL_Z              = 0.041
BOWL_FOOTPRINT      = 0.15   # radius of bowl exclusion zone for fruits
BOWL_MARGIN         = 0.06   # extra clearance around footprint

# ── Task configuration ────────────────────────────────────────────────────────

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim: int            = 1500
    use_adaptive_grasp: bool   = True
    use_force_grasp: bool      = False
    force_fast: float        = 50.0
    force_slow: float        = 20.0
    n_objects: int           = 3
    grasp_axis_angle_std: float  = 0.15   # rad — rotation noise on close axis
    grasp_pos_xy_std: float      = 0.008  # m   — XY noise on grasp point
    grasp_pos_z_std: float       = 0.004  # m   — Z noise on grasp point
    lift_z_range: tuple          = (0.18, 0.28)  # m — uniform lift height
    drop_xy_std: float           = 0.015  # m   — XY noise on above-bowl waypoint
    drop_z_range: tuple          = (0.08, 0.15)

class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal['collect', 'eval'] = 'collect',
        render_mode: str | None = None,
        **kwargs,
    ):

        cfg.uipc_sim.contact.default_friction_ratio = 0.5

        super().__init__(cfg, mode, render_mode, **kwargs)

    # ── Actor creation (called once at env build) ─────────────────────────────
    def _sample_grasp_noise(self) -> dict:
        """
        Pre-sample all stochastic perturbations for one episode step.
        Centralising them here makes logging and reproducibility easy.

        Returns a dict consumed by _build_grasp_pose and _play_once.
        """
        cfg = self.cfg

        # ── Close-axis rotation noise ─────────────────────────────────────────
        # Small rotation around Z (world-up) applied to base_close_axis.
        # Using a Gaussian clamped to ±2σ to avoid degenerate orientations.
        axis_angle = float(
            np.clip(
                self.rng.normal(0.0, cfg.grasp_axis_angle_std),
                -2 * cfg.grasp_axis_angle_std,
                +2 * cfg.grasp_axis_angle_std,
            )
        )
        ca, sa = np.cos(axis_angle), np.sin(axis_angle)
        # 2-D rotation around Z — applied later in _build_grasp_pose
        axis_rot_z = np.array([[ca, -sa, 0.],
                                [sa,  ca, 0.],
                                [0.,  0., 1.]])

        # ── Grasp position noise ──────────────────────────────────────────────
        grasp_pos_noise = np.array([
            float(self.rng.normal(0.0, cfg.grasp_pos_xy_std)),
            float(self.rng.normal(0.0, cfg.grasp_pos_xy_std)),
            float(self.rng.normal(0.0, cfg.grasp_pos_z_std)),
        ])

        # ── Lift height ───────────────────────────────────────────────────────
        lift_z = float(self.rng.uniform(*cfg.lift_z_range))

        # ── Drop waypoint noise ───────────────────────────────────────────────
        drop_xy_noise = np.array([
            float(self.rng.normal(0.0, cfg.drop_xy_std)),
            float(self.rng.normal(0.0, cfg.drop_xy_std)),
        ])
        drop_z = float(self.rng.uniform(*cfg.drop_z_range))

        noise = {
            'axis_rot_z':    axis_rot_z,
            'grasp_pos':     grasp_pos_noise,
            'lift_z':        lift_z,
            'drop_xy':       drop_xy_noise,
            'drop_z':        drop_z,
        }

        self.domain_rand_params.update({
        'grasp_axis_angle_deg': float(np.degrees(axis_angle)),
        'grasp_pos_noise_x':    float(grasp_pos_noise[0]),
        'grasp_pos_noise_y':    float(grasp_pos_noise[1]),
        'grasp_pos_noise_z':    float(grasp_pos_noise[2]),
        'lift_z':               lift_z,
        'drop_xy_noise_x':      float(drop_xy_noise[0]),
        'drop_xy_noise_y':      float(drop_xy_noise[1]),
        'drop_z':               drop_z,
        })

        print(
            f"[noise] axis_angle={np.degrees(axis_angle):.1f}° | "
            f"grasp_pos={grasp_pos_noise.round(4)} | "
            f"lift_z={lift_z:.3f} | "
            f"drop_xy={drop_xy_noise.round(4)} | "
            f"drop_z={drop_z:.3f}"
        )
        return noise
    def create_actors(self):
        """Spawn all fruit actors off-table and the bowl at its base pose."""
        self._fruits: dict[str, Actor] = {}

        for i, name in enumerate(ALL_FRUIT_NAMES):
            offscreen_pose = Pose(
                [OFFSCREEN_X, OFFSCREEN_Y_START - i * GRID_SPACING_Y, OFFSCREEN_Z],
                [1, 0, 0, 0],
            )
            UipcObjectCfg.AffineBodyConstitutionCfg()
            actor = self._actor_manager.add_from_usd_file(
                name=name,
                asset_path=FRUITS_CONFIGS[name]['asset_path'],
                pose=offscreen_pose,
                density=FRUITS_CONFIGS[name]['density'],
            )
            self._fruits[name] = actor

        self._bowl = self._actor_manager.add_from_usd_file(
            name='bowl',
            asset_path=BOWL_ASSET_PATH,
            pose=BOWL_BASE_POSE,
        )

    # ── Random non-overlapping spawn poses ───────────────────────────────────
    def _random_init_poses(self, n: int, min_dist: float = 0.15) -> list:
        MAX_ATTEMPTS = 200

        # Table spawn bounds
        TABLE_X_MIN, TABLE_X_MAX = 0.20, 0.70
        TABLE_Y_MIN, TABLE_Y_MAX = -0.4, 0.4

        # Exclusion zone dynamically built from randomized bowl pose
        bowl_x, bowl_y = self._bowl_pose.p[0], self._bowl_pose.p[1]
        excl_r = BOWL_FOOTPRINT + BOWL_MARGIN  # total exclusion radius

        def _in_bowl_zone(x: float, y: float) -> bool:
            return np.sqrt((x - bowl_x)**2 + (y - bowl_y)**2) < excl_r

        poses = []
        while len(poses) < n:
            for _ in range(MAX_ATTEMPTS):
                x  = self.rng.uniform(TABLE_X_MIN, TABLE_X_MAX)
                y  = self.rng.uniform(TABLE_Y_MIN, TABLE_Y_MAX)
                rz = self.rng.uniform(0.0, 2 * np.pi)

                if _in_bowl_zone(x, y):
                    continue

                q = np.array([np.cos(rz / 2), 0.0, 0.0, np.sin(rz / 2)])
                candidate = Pose([x, y, TABLE_SPAWN_Z], q)

                too_close = any(
                    np.linalg.norm(candidate.p[:2] - p.p[:2]) < min_dist
                    for p in poses
                )
                if not too_close:
                    poses.append(candidate)
                    break
            else:
                raise RuntimeError(
                    f"Could not place object {len(poses)+1}/{n} after "
                    f"{MAX_ATTEMPTS} attempts — reduce n or min_dist."
                )
        return poses

    # ── Per-episode reset ─────────────────────────────────────────────────────
    def _reset_actors(self):
        self.success_type: str = 'none'  
        self._robot_manager._reset_idx()

        # 1. Randomize bowl pose (kept near table centre)
        bowl_x = float(self.rng.uniform(BOWL_X_MIN, BOWL_X_MAX))
        bowl_y = float(self.rng.uniform(BOWL_Y_MIN, BOWL_Y_MAX))
        self._bowl_pose = Pose(
            [bowl_x, bowl_y, BOWL_Z],
            BOWL_BASE_POSE.q,
        )
        self._bowl.set_pose(self._bowl_pose)

        # 2. Decide how many fruits appear this episode
        if len(ALL_FRUIT_NAMES) <= 2 :
            n = len(ALL_FRUIT_NAMES)
        else :
            n = int(self.rng.integers(2, len(ALL_FRUIT_NAMES)))

        # 3. Choose which fruits are active
        chosen_indices = self.rng.choice(len(ALL_FRUIT_NAMES), size=n, replace=False)
        self._active_names: list[str] = [ALL_FRUIT_NAMES[i] for i in chosen_indices]

        # 4. Choose target fruit and sample an instruction
        target_idx = int(self.rng.integers(0, n))
        self._target_name: str = self._active_names[target_idx]
        instruction_pool = FRUITS_CONFIGS[self._target_name]['instructions']
        self.instruction = str(self.rng.choice(instruction_pool))

        # 5. Compute final table poses (exclusion zone now tracks randomized bowl)
        final_poses = self._random_init_poses(n, min_dist=0.15)

        # 6. Park ALL fruits off-table first
        for i, name in enumerate(ALL_FRUIT_NAMES):
            self._fruits[name].set_pose(
                Pose(
                    [OFFSCREEN_X, OFFSCREEN_Y_START - i * GRID_SPACING_Y, OFFSCREEN_Z],
                    [1, 0, 0, 0],
                )
            )

        # 7. Teleport active fruits to safe height above final XY
        for idx, name in enumerate(self._active_names):
            p = final_poses[idx]
            self._fruits[name].set_pose(
                Pose([p.p[0], p.p[1], SPAWN_SAFE_Z], p.q)
            )

        # 8. Lower to final resting pose
        for idx, name in enumerate(self._active_names):
            self._fruits[name].set_pose(final_poses[idx])

        # 9. Bookkeeping
        self._success_steps: int = 0
        self.domain_rand_params['active_fruits'] = self._active_names
        self.domain_rand_params['target_fruit']  = self._target_name
        self.domain_rand_params['instruction']   = self.instruction
        self.domain_rand_params['bowl_pos']      = [bowl_x, bowl_y]

        print(
            f"[puts_fruits_bowl] episode reset | "
            f"active={self._active_names} | target='{self._target_name}' | "
            f"bowl=({bowl_x:.3f}, {bowl_y:.3f}) | "
            f"instruction='{self.instruction}'"
        )

    # ── Grasp pose builder ────────────────────────────────────────────────────

    def _build_grasp_pose(self, noise: dict) -> tuple:
        target = self._fruits[self._target_name]
        cfg    = FRUITS_CONFIGS[self._target_name]
        obj_p  = target.get_pose().p

        if self._target_name == 'banana':
            base_close_axis = np.array([-sqrt(2)/2, sqrt(2)/2, 0.0])
        else:
            base_close_axis = np.array([1.0, 0.0, 0.0])

        # ── 1. Apply axis noise (rotation around Z) ───────────────────────────
        base_close_axis = noise['axis_rot_z'] @ base_close_axis

        # ── 2. Rotate by object pose ──────────────────────────────────────────
        q = self._fruits[self._target_name].get_pose().q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        if self._target_name == 'pear' :
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            cy, sy = np.cos(yaw), np.sin(yaw)
            R = np.array([
                [cy, -sy, 0.],
                [sy,  cy, 0.],
                [0.,  0., 1.],
            ])
        else :
            R = np.array([
                [1-2*(y*y+z*z),  2*(x*y-z*w),  2*(x*z+y*w)],
                [  2*(x*y+z*w),1-2*(x*x+z*z),  2*(y*z-x*w)],
                [  2*(x*z-y*w),  2*(y*z+x*w),1-2*(x*x+y*y)],
            ])
        close_axis = R @ base_close_axis

        # ── 3. Grasp position with noise ──────────────────────────────────────
        grasp_p = (
            obj_p
            + np.array([0.0, 0.0, cfg['grasp_z']])
            + noise['grasp_pos']                    # ← XYZ perturbation
        )

        grasp_pose  = construct_grasp_pose(grasp_p, np.array([0.0, 0.0, 1.0]), close_axis)
        contact_idx = target.register_point(pose=grasp_pose, type='contact')
        return grasp_pose, contact_idx

    # ── Pre-move ──────────────────────────────────────────────────────────────

    def pre_move(self):
        """Open gripper before the episode begins."""
        self.move(self.atom.open_gripper(pos=1,force=self.cfg.force_fast))

    def check_mid_failure(self) -> bool:
        """Return True if the robot lost the fruit after grasping."""
        pose_object = self._fruits[self._target_name].get_pose().p
        pose_ee     = self._robot_manager.get_gripper_center_pose().p
        lost = np.linalg.norm(pose_object - pose_ee) > 0.15
        if lost:
            print(f"[puts_fruits_bowl] mid-failure detected: fruit too far from EE")
        return lost

    # ── Main episode logic ────────────────────────────────────────────────────
    # ── Success criterion ─────────────────────────────────────────────────────

    def check_success(self) -> bool:
        """
        The episode succeeds when the target fruit is:
          - horizontally within _SUCCESS_XY_TOL of the bowl centre, AND
          - vertically below bowl_z + _SUCCESS_Z_MIN  (i.e. inside the bowl).
        """
        target   = self._fruits[self._target_name]
        bowl_p   = self._bowl.get_pose().p
        obj_p    = target.get_pose().p

        xy_err = np.linalg.norm(obj_p[:2] - bowl_p[:2])
        xy_ok  = xy_err < _SUCCESS_XY_TOL
        z_ok   = obj_p[2] < bowl_p[2] + _SUCCESS_Z_MIN

        in_bowl = xy_ok and z_ok
        return in_bowl
    
    def _play_once(self):
        """Grasp the target fruit and drop it into the bowl."""
        cfg   = FRUITS_CONFIGS[self._target_name]

        # Sample all noise once per episode so every step uses the same draw
        noise = self._sample_grasp_noise()
        grasp_pose, contact_idx = self._build_grasp_pose(noise)
        obj_p  = self._fruits[self._target_name].get_pose().p
        ee_now = self.atom.get_arm_pose()
        approach_z = 0.25

        # ── 2. Go above fruit (position only, keep current orientation) ────────────
        above_fruit_pos = np.array([
            obj_p[0],
            obj_p[1],
            obj_p[2] + cfg['grasp_z'] + approach_z
        ])

        above_fruit = Pose(above_fruit_pos, ee_now.q)

        self.move(
            self.atom.move_to_pose(above_fruit),
        )

        # ── 3. Rotate wrist in place (IMPORTANT) ───────────────────────────────────
        above_fruit_rotated = Pose(above_fruit_pos, grasp_pose.q)

        self.move(
            self.atom.move_to_pose(above_fruit_rotated),
        )

        # ── 4. Descend straight down (no rotation anymore) ─────────────────────────
        self.move(self.atom.grasp_actor(
            self._fruits[self._target_name],
            contact_point_id=contact_idx,
            is_close=False,
        ))
        force_to_apply = self.cfg.force_slow if self._target_name in  [''] else self.cfg.force_fast
        self.move(self.atom.close_gripper(force=force_to_apply,depth_threshold=FRUITS_CONFIGS[self._target_name]['depth_threshold']))
        # ── 5. Lift — randomised height ───────────────────────────────────────
        self.move(self.atom.move_by_displacement(z=noise['lift_z']))
        
        if self.check_mid_failure():
            print("[puts_fruits_bowl] aborting: fruit dropped during lift")
            return

        # ── 6. Move above bowl — with XY noise ────────────────────────────────
        bowl_p = self._bowl.get_pose().p
        ee_now = self.atom.get_arm_pose()

        above_bowl = Pose(
            [
                bowl_p[0] + noise['drop_xy'][0],   # ← XY noise on target waypoint
                bowl_p[1] + noise['drop_xy'][1],
                ee_now.p[2],
            ],
            ee_now.q,
        )
        self.move(self.atom.move_to_pose(above_bowl), time_dilation_factor=0.5)

        # ── 7. Lower — randomised descent depth ──────────────────────────────
        self.move(
            self.atom.move_by_displacement(z=-noise['drop_z']),
            time_dilation_factor=0.5,
        )

        if self.check_success():
            self.success_type = 'slip_success' 

        # ── 8. Release ────────────────────────────────────────────────────────
        self.move(self.atom.open_gripper(force=self.cfg.force_fast, steps=40))

        self.delay(10)
        if self.check_success() and self.success_type != 'slip_success':
            self.success_type = 'normal_success' 

