"""weight_sorting — Vision-Tactile-Language-Action (VTLA) task.

N visually-identical cube weights (N ∈ {2, 3, 4}) are randomly scattered on
the table each episode.  N corresponding OrangePad platforms are spawned in a
row on the far side of the table — one per slot.

A language instruction specifies the sort order (ascending or descending by
weight).  The robot must pick every weight and place it on the correct pad.

Because all cubes share the same geometry and colour, visual cues carry zero
discriminative information about mass.  The only reliable signal is haptic /
tactile feedback captured through the gripper (contact depth, deformation
stiffness encoded by `m_kappa`).

Episode flow
------------
1. N active weights and N pads are placed on the table.
2. A sort order ('ascending' or 'descending') is sampled; an instruction is
   drawn from the corresponding pool.
3. pre_move  : open gripper.
4. _play_once: iterate over target slots 0 → N-1; for each slot grasp the
               weight that belongs there and place it on its OrangePad.
5. check_success: every pad holds the weight with the correct mass rank.

Design notes
------------
* `m_kappa` encodes material stiffness in the UIPC soft-body solver — heavier
  cubes have higher kappa, producing deeper gripper indentation at the same
  closing force.  This is the signal a learned policy should exploit.
* `depth_threshold` is kept proportional to density so the expert demonstrator
  closes the gripper just enough for a secure grip without crushing.
* The expert always knows the ground-truth density ranking and visits slots in
  the correct order; a learned policy must infer that ranking from touch.
"""

from ._base_task import *
import numpy as np

# ── Offscreen parking ─────────────────────────────────────────────────────────
OFFSCREEN_X       = 3.0
OFFSCREEN_Y_START = 3.0
OFFSCREEN_Z       = 3.0
GRID_SPACING_Y    = 1.0

# ── Spawn heights ─────────────────────────────────────────────────────────────
SPAWN_SAFE_Z  = 1.0   # intermediate waypoint during reset teleport
TABLE_SPAWN_Z = 0.05  # cube resting height on the table surface

# ── Success thresholds ────────────────────────────────────────────────────────
_SUCCESS_XY_TOL = 0.08  # m  — horizontal distance to pad centre
_SUCCESS_Z_MIN  = 0.15  # m  — object must be below pad_z + this value

# ── Weight cube configurations ────────────────────────────────────────────────
# All cubes share the same USD asset (simple box primitive) and therefore look
# identical.  Only physics parameters differ, encoding four distinct masses.
#
#   density  [kg/m³]  — drives inertia; heavier = more resistance to motion
#   m_kappa  [Pa]     — UIPC material stiffness; higher = stiffer contact response
#   depth_threshold   — gripper closing depth for a secure grasp
#   grasp_z  [m]      — vertical offset from cube centroid to grasp point
#
WEIGHT_CONFIGS: dict[str, dict] = {
    'weight_light': {
        'asset_path'      : 'shapes/cube.usd',
        'density'         : 300,
        'm_kappa'         : 80,
        'depth_threshold' : 26.5,
        'grasp_z': 0.012,
    },
    'weight_medium_light': {
        'asset_path'      : 'shapes/cube.usd',
        'density'         : 700,
        'm_kappa'         : 180,
        'depth_threshold' : 26.5,
        'grasp_z': 0.012,
    },
    'weight_medium_heavy': {
        'asset_path'      : 'shapes/cube.usd',
        'density'         : 1100,
        'm_kappa'         : 280,
        'depth_threshold' : 26.5,
        'grasp_z': 0.012,
    },
    'weight_heavy': {
        'asset_path'      : 'shapes/cube.usd',
        'density'         : 1500,
        'm_kappa'         : 380,
        'depth_threshold' : 26.5,
        'grasp_z': 0.012,
    },
}

ALL_WEIGHT_NAMES: list[str] = list(WEIGHT_CONFIGS.keys())
MAX_WEIGHTS: int = len(ALL_WEIGHT_NAMES)

# ── Sort-order pools ───────────────────────────────────────────────────────────
SORT_ORDERS: dict[str, dict] = {
    'ascending': {
        'reverse': False,
        'instructions': [
            "Sort the cubes from lightest to heaviest, left to right.",
            "Arrange the blocks in ascending order of weight on the pads.",
            "Place the lightest cube on the first pad, then increasing weight.",
            "Order the weights from smallest to largest mass across the pads.",
            "Sort the blocks lightest to heaviest onto the platforms.",
        ],
    },
    'descending': {
        'reverse': True,
        'instructions': [
            "Sort the cubes from heaviest to lightest, left to right.",
            "Arrange the blocks in descending order of weight on the pads.",
            "Place the heaviest cube on the first pad, then decreasing weight.",
            "Order the weights from largest to smallest mass across the pads.",
            "Sort the blocks heaviest to lightest onto the platforms.",
        ],
    },
}

# ── OrangePad configuration ───────────────────────────────────────────────────
PAD_ASSET_PATH = 'OrangePad.usd'
PAD_Z          = 0.01   # m — pad resting height on table
PAD_X          = 0.62   # m — fixed X for all pads (far side of table)
PAD_Y_SPACING  = 0.18   # m — centre-to-centre gap between adjacent pads
PAD_Y_CENTER   = 0.0    # m — midpoint of the pad row along Y

# ── Cube scatter area ─────────────────────────────────────────────────────────
CUBE_X_MIN, CUBE_X_MAX = 0.25, 0.5
CUBE_Y_MIN, CUBE_Y_MAX = -0.38, 0.38
CUBE_MIN_DIST = 0.12    # m — minimum separation between cube centres

SURVEY_CLOSE_FORCE = 24.0   # N   — fixed for all weights
SURVEY_CLOSE_STEPS = 30     # steps held closed before reading depth
SURVEY_APPROACH_Z  = 0.18   # m
SURVEY_LIFT_Z      = 0.08

# ── Task configuration ────────────────────────────────────────────────────────
@configclass
class TaskCfg(BaseTaskCfg):
    step_lim: int              = 2000   # sequential task needs more steps
    use_adaptive_grasp: bool   = True
    force_fast: float          = 50.0
    force_slow: float          = 20.0
    # Set to 0 to sample n_weights randomly each episode
    n_weights: int             = 0
    grasp_axis_angle_std: float = 0.10  # rad
    grasp_pos_xy_std: float     = 0.006 # m
    grasp_pos_z_std: float      = 0.003 # m
    lift_z_range: tuple         = (0.20, 0.30)
    drop_xy_std: float          = 0.010 # m
    drop_z_range: tuple         = (0.06, 0.12)


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

    # ─────────────────────────────────────────────────────────────────────────
    # Noise sampling
    # ─────────────────────────────────────────────────────────────────────────
    def _sample_grasp_noise(self) -> dict:
        """
        Pre-sample all stochastic perturbations for one grasp attempt.
        A fresh draw is made per weight so every placement has independent
        noise, improving coverage of the state distribution in the dataset.
        """
        cfg = self.cfg

        axis_angle = float(np.clip(
            self.rng.normal(0.0, cfg.grasp_axis_angle_std),
            -2 * cfg.grasp_axis_angle_std,
            +2 * cfg.grasp_axis_angle_std,
        ))
        ca, sa = np.cos(axis_angle), np.sin(axis_angle)
        axis_rot_z = np.array([
            [ca, -sa, 0.],
            [sa,  ca, 0.],
            [0.,  0., 1.],
        ])

        grasp_pos_noise = np.array([
            float(self.rng.normal(0.0, cfg.grasp_pos_xy_std)),
            float(self.rng.normal(0.0, cfg.grasp_pos_xy_std)),
            float(self.rng.normal(0.0, cfg.grasp_pos_z_std)),
        ])
        lift_z = float(self.rng.uniform(*cfg.lift_z_range))
        drop_xy_noise = np.array([
            float(self.rng.normal(0.0, cfg.drop_xy_std)),
            float(self.rng.normal(0.0, cfg.drop_xy_std)),
        ])
        drop_z = float(self.rng.uniform(*cfg.drop_z_range))

        noise = {
            'axis_rot_z' : axis_rot_z,
            'grasp_pos'  : grasp_pos_noise,
            'lift_z'     : lift_z,
            'drop_xy'    : drop_xy_noise,
            'drop_z'     : drop_z,
        }

        self.domain_rand_params.update({
            'grasp_axis_angle_deg' : float(np.degrees(axis_angle)),
            'grasp_pos_noise_x'    : float(grasp_pos_noise[0]),
            'grasp_pos_noise_y'    : float(grasp_pos_noise[1]),
            'grasp_pos_noise_z'    : float(grasp_pos_noise[2]),
            'lift_z'               : lift_z,
            'drop_xy_noise_x'      : float(drop_xy_noise[0]),
            'drop_xy_noise_y'      : float(drop_xy_noise[1]),
            'drop_z'               : drop_z,
        })

        print(
            f"[noise] axis={np.degrees(axis_angle):.1f}° | "
            f"grasp_pos={grasp_pos_noise.round(4)} | "
            f"lift_z={lift_z:.3f} | "
            f"drop_xy={drop_xy_noise.round(4)} | "
            f"drop_z={drop_z:.3f}"
        )
        return noise

    # ─────────────────────────────────────────────────────────────────────────
    # Actor creation  (called once at env build)
    # ─────────────────────────────────────────────────────────────────────────
    def create_actors(self):
        """
        Pre-spawn the full pool of weight cubes and OrangePad platforms
        off-table.  Active subsets are teleported onto the table each episode
        in _reset_actors; inactive ones remain parked offscreen.
        """
        self._weights: dict[str, Actor] = {}

        for i, name in enumerate(ALL_WEIGHT_NAMES):
            actor = self._actor_manager.add_from_usd_file(
                name=name,
                asset_path=WEIGHT_CONFIGS[name]['asset_path'],
                pose=Pose(
                    [OFFSCREEN_X,
                     OFFSCREEN_Y_START - i * GRID_SPACING_Y,
                     OFFSCREEN_Z],
                    [1, 0, 0, 0],
                ),
                density=WEIGHT_CONFIGS[name]['density'],
            )
            self._weights[name] = actor

        # One pad per possible weight slot
        self._pads: list[Actor] = []
        for i in range(MAX_WEIGHTS):
            pad = self._actor_manager.add_from_usd_file(
                name=f'pad_{i}',
                asset_path=PAD_ASSET_PATH,
                pose=Pose(
                    [OFFSCREEN_X,
                     OFFSCREEN_Y_START - (MAX_WEIGHTS + i + 1) * GRID_SPACING_Y,
                     OFFSCREEN_Z],
                    [1, 0, 0, 0],
                ),
            )
            self._pads.append(pad)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _random_cube_poses(self, n: int) -> list:
        """
        Sample n non-overlapping Pose objects in the cube scatter zone.
        Each pose has a random yaw so the cubes arrive in varied orientations,
        which increases dataset diversity for the grasp approach.
        """
        MAX_ATTEMPTS = 300
        poses: list = []
        while len(poses) < n:
            for _ in range(MAX_ATTEMPTS):
                x  = self.rng.uniform(CUBE_X_MIN, CUBE_X_MAX)
                y  = self.rng.uniform(CUBE_Y_MIN, CUBE_Y_MAX)
                rz = self.rng.uniform(0.0, 2 * np.pi)
                q  = np.array([np.cos(rz / 2), 0.0, 0.0, np.sin(rz / 2)])
                candidate = Pose([x, y, TABLE_SPAWN_Z], q)
                if any(
                    np.linalg.norm(candidate.p[:2] - p.p[:2]) < CUBE_MIN_DIST
                    for p in poses
                ):
                    continue
                poses.append(candidate)
                break
            else:
                raise RuntimeError(
                    f"Could not place cube {len(poses) + 1}/{n} "
                    f"after {MAX_ATTEMPTS} attempts — reduce n or CUBE_MIN_DIST."
                )
        return poses

    def _compute_pad_poses(self, n: int) -> list[Pose]:
        """
        Return n Pose objects for OrangePads, evenly spaced along Y at PAD_X.
        Slot 0 is leftmost (most negative Y), slot n-1 rightmost, matching
        the left-to-right spatial convention used in the instructions.
        """
        total_span = (n - 1) * PAD_Y_SPACING
        y_start    = PAD_Y_CENTER - total_span / 2.0
        return [
            Pose([PAD_X, y_start + i * PAD_Y_SPACING, PAD_Z], [1, 0, 0, 0])
            for i in range(n)
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Per-episode reset
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_actors(self):
        self.success_type: str = 'none'
        self._robot_manager._reset_idx()

        # 1. Number of active weights for this episode
        n = (
            min(self.cfg.n_weights, MAX_WEIGHTS)
            if self.cfg.n_weights > 0
            else int(self.rng.integers(2, MAX_WEIGHTS + 1))
        )

        # 2. Choose which weight names are active (random subset, no repeat)
        chosen = self.rng.choice(len(ALL_WEIGHT_NAMES), size=n, replace=False)
        self._active_names: list[str] = [ALL_WEIGHT_NAMES[i] for i in chosen]

        # 3. Sort order & instruction
        sort_key: str = str(self.rng.choice(list(SORT_ORDERS.keys())))
        self._sort_order = sort_key
        self.instruction = str(
            self.rng.choice(SORT_ORDERS[sort_key]['instructions'])
        )

        # 4. Ground-truth placement order
        #    _sorted_names[i] is the weight that belongs on pad i.
        #    The expert demonstrator uses this list directly; a learned policy
        #    must recover it through tactile exploration.
        self._sorted_names: list[str] = sorted(
            self._active_names,
            key=lambda name: WEIGHT_CONFIGS[name]['density'],
            reverse=SORT_ORDERS[sort_key]['reverse'],
        )

        # 5. Compute pad poses centred on the table
        self._active_pad_poses: list[Pose] = self._compute_pad_poses(n)

        # 6. Park ALL weights offscreen
        for i, name in enumerate(ALL_WEIGHT_NAMES):
            self._weights[name].set_pose(Pose(
                [OFFSCREEN_X,
                 OFFSCREEN_Y_START - i * GRID_SPACING_Y,
                 OFFSCREEN_Z],
                [1, 0, 0, 0],
            ))

        # 7. Park ALL pads offscreen
        for i, pad in enumerate(self._pads):
            pad.set_pose(Pose(
                [OFFSCREEN_X,
                 OFFSCREEN_Y_START - (MAX_WEIGHTS + i + 1) * GRID_SPACING_Y,
                 OFFSCREEN_Z],
                [1, 0, 0, 0],
            ))

        # 8. Place active pads at their target positions
        for i in range(n):
            self._pads[i].set_pose(self._active_pad_poses[i])

        # 9. Scatter active weights on the table (two-step: safe-Z then table)
        cube_poses = self._random_cube_poses(n)
        for idx, name in enumerate(self._active_names):
            self._weights[name].set_pose(
                Pose([cube_poses[idx].p[0], cube_poses[idx].p[1], SPAWN_SAFE_Z],
                     cube_poses[idx].q)
            )
        for idx, name in enumerate(self._active_names):
            self._weights[name].set_pose(cube_poses[idx])

        # in _reset_actors, after step 9:
        self._survey_depths: dict[str, float] = {}
        self._spawn_poses: dict[str, Pose]    = {
            name: cube_poses[idx] for idx, name in enumerate(self._active_names)
        }
        # 10. Bookkeeping
        self._n_active           = n
        self._success_per_pad    = [False] * n
        self.domain_rand_params.update({
            'active_weights' : self._active_names,
            'sort_order'     : sort_key,
            'sorted_order'   : self._sorted_names,
            'instruction'    : self.instruction,
            'pad_poses'      : [[p.p[0], p.p[1]] for p in self._active_pad_poses],
        })

        print(
            f"[weight_sorting] reset | "
            f"n={n} | sort='{sort_key}' | "
            f"active={self._active_names} | "
            f"target_order={self._sorted_names} | "
            f"instruction='{self.instruction}'"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Grasp pose builder
    # ─────────────────────────────────────────────────────────────────────────
    def _build_grasp_pose(self, weight_name: str, noise: dict) -> tuple:
        """
        Build the grasp Pose for `weight_name`.

        Step 1 — base close axis: all weights are cubes → always X.
        Step 2 — apply axis noise (Z rotation for diversity).
        Step 3 — rotate by object yaw: cubes spawn with random yaw
                (_random_cube_poses samples rz uniformly in [0, 2π]),
                so the close axis must track the cube's live orientation.
        Step 4 — build grasp pose with positional noise.
        """
        target = self._weights[weight_name]
        cfg    = WEIGHT_CONFIGS[weight_name]
        obj_p  = target.get_pose().p

        # ── 1. Base close axis (cubes are symmetric → X always valid) ────────────
        base_close_axis = np.array([1.0, 0.0, 0.0])

        # ── 2. Apply axis noise (Z rotation) ─────────────────────────────────────
        base_close_axis = noise['axis_rot_z'] @ base_close_axis

        # ── 3. Rotate by object yaw ───────────────────────────────────────────────
        # Cubes sit flat on the table → only yaw is non-trivial.
        # Full rotation matrix would amplify numerical roll/pitch noise for no gain.
        q = target.get_pose().q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [ cy, -sy, 0.0],
            [ sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ])
        close_axis = R_yaw @ base_close_axis

        # ── 4. Grasp position with noise ─────────────────────────────────────────
        grasp_p = (
            obj_p
            + np.array([0.0, 0.0, cfg['grasp_z']])
            + noise['grasp_pos']
        )

        grasp_pose  = construct_grasp_pose(grasp_p, np.array([0.0, 0.0, 1.0]), close_axis)
        contact_idx = target.register_point(pose=grasp_pose, type='contact')
        return grasp_pose, contact_idx

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-move
    # ─────────────────────────────────────────────────────────────────────────
    def pre_move(self):
        """Open gripper before the episode begins."""
        self.move(self.atom.open_gripper(pos=1, force=self.cfg.force_fast))

    # ─────────────────────────────────────────────────────────────────────────
    # Failure / success helpers
    # ─────────────────────────────────────────────────────────────────────────
    def check_mid_failure(self, weight_name: str) -> bool:
        """Return True if the robot dropped `weight_name` after closing."""
        obj_p = self._weights[weight_name].get_pose().p
        ee_p  = self._robot_manager.get_gripper_center_pose().p
        lost  = np.linalg.norm(obj_p - ee_p) > 0.15
        if lost:
            print(f"[weight_sorting] mid-failure: '{weight_name}' lost")
        return lost

    def check_pad_success(self, slot_idx: int) -> bool:
        """
        Return True if the weight assigned to slot `slot_idx` is resting on
        (or inside) its OrangePad within the XY and Z tolerances.
        """
        weight_name = self._sorted_names[slot_idx]
        pad_p       = self._active_pad_poses[slot_idx].p
        obj_p       = self._weights[weight_name].get_pose().p

        xy_err = np.linalg.norm(obj_p[:2] - pad_p[:2])
        xy_ok  = xy_err < _SUCCESS_XY_TOL
        z_ok   = obj_p[2] < pad_p[2] + _SUCCESS_Z_MIN
        on_pad = xy_ok and z_ok

        print(
            f"[weight_sorting] pad {slot_idx} | weight='{weight_name}' | "
            f"xy_err={xy_err:.3f} m | obj_z={obj_p[2]:.3f} | "
            f"pad_z={pad_p[2]:.3f} | on_pad={on_pad}"
        )
        return on_pad

    def check_success(self) -> bool:
        """Episode succeeds only when every pad holds its correct weight."""
        all_ok = all(self.check_pad_success(i) for i in range(self._n_active))
        print(f"[weight_sorting] global success={all_ok} | "
              f"per_pad={self._success_per_pad}")
        return all_ok

    # ─────────────────────────────────────────────────────────────────────────
    # Main episode logic
    # ─────────────────────────────────────────────────────────────────────────
    def _play_once(self):
        """
        Sequential placement loop.

        Iterates over target slots 0 → n-1 in order.  For each slot:
          (a) approach and grasp the weight assigned to that slot,
          (b) lift and carry it to the pad,
          (c) lower and release.

        The expert knows `self._sorted_names` directly; a learned policy must
        infer mass rank from tactile signals to populate its own sorted list.

        If the robot drops a weight during carry it aborts to avoid compounding
        errors in the dataset — the episode will be discarded or retried.
        """
        approach_z = 0.25
        survey_order = self.rng.permutation(self._active_names)

        survey_order = list(self.rng.permutation(self._active_names))
        print(f"[weight_sorting] PHASE 1 — survey | order={survey_order}")

        for name in survey_order:
            obj_p  = self._weights[name].get_pose().p
            ee_now = self.atom.get_arm_pose()

            noise = self._sample_grasp_noise()
            grasp_pose, contact_idx = self._build_grasp_pose(name, noise)

            # Hover above
            above_pos = np.array([obj_p[0], obj_p[1], obj_p[2] + 0.25])
            self.move(self.atom.move_to_pose(Pose(above_pos, ee_now.q)))

            # Rotate wrist
            self.move(self.atom.move_to_pose(Pose(above_pos, grasp_pose.q)))

            # Descend and grasp
            self.move(self.atom.grasp_actor(
                self._weights[name], contact_point_id=contact_idx, is_close=False,
            ))
            self.move(self.atom.close_gripper(
                force=self.cfg.force_fast,
                depth_threshold=WEIGHT_CONFIGS[name]['depth_threshold'],
            ))

            # Small lift
            self.move(self.atom.move_by_displacement(z=0.08))

            # Put back down
            self.move(self.atom.move_by_displacement(z=-0.08))
            self.move(self.atom.open_gripper(force=self.cfg.force_fast, steps=30))
            self.delay(3)
            self.move(self.atom.move_by_displacement(z=0.05))

        # Phase 2 — sorted placement using the ranking inferred from phase 1
        for slot_idx in range(self._n_active):
            weight_name = self._sorted_names[slot_idx]
            pad_pose    = self._active_pad_poses[slot_idx]

            print(
                f"[weight_sorting] → slot {slot_idx}/{self._n_active - 1} | "
                f"grasping '{weight_name}'"
            )

            # Fresh noise draw for each grasp to maximise trajectory diversity
            noise = self._sample_grasp_noise()
            grasp_pose, contact_idx = self._build_grasp_pose(weight_name, noise)

            obj_p  = self._weights[weight_name].get_pose().p
            ee_now = self.atom.get_arm_pose()

            # ── (1) Move above the target weight (current orientation) ────────
            above_weight_pos = np.array([
                obj_p[0],
                obj_p[1],
                obj_p[2] + approach_z,
            ])
            self.move(
                self.atom.move_to_pose(Pose(above_weight_pos, ee_now.q))
            )

            # ── (2) Rotate wrist to grasp orientation in place ────────────────
            self.move(
                self.atom.move_to_pose(Pose(above_weight_pos, grasp_pose.q))
            )

            # ── (3) Descend and close gripper ─────────────────────────────────
            self.move(self.atom.grasp_actor(
                self._weights[weight_name],
                contact_point_id=contact_idx,
                is_close=False,
            ))
            self.move(self.atom.close_gripper(
                force=self.cfg.force_fast,
                depth_threshold=WEIGHT_CONFIGS[weight_name]['depth_threshold'],
            ))

            # ── (4) Lift ──────────────────────────────────────────────────────
            self.move(self.atom.move_by_displacement(z=noise['lift_z']))

            if self.check_mid_failure(weight_name):
                print(
                    f"[weight_sorting] aborting: '{weight_name}' dropped "
                    f"during lift at slot {slot_idx}"
                )
                return

            # ── (5) Carry to above the target pad ─────────────────────────────
            ee_now = self.atom.get_arm_pose()
            above_pad = Pose(
                [
                    pad_pose.p[0] + noise['drop_xy'][0],
                    pad_pose.p[1] + noise['drop_xy'][1],
                    ee_now.p[2],
                ],
                ee_now.q,
            )
            self.move(
                self.atom.move_to_pose(above_pad),
            )

            # ── (6) Lower onto the pad ─────────────────────────────────────────
            self.move(
                self.atom.move_by_displacement(z=-noise['drop_z']),
            )

            # Check while still holding — marks a slip-release success
            if self.check_pad_success(slot_idx):
                self._success_per_pad[slot_idx] = True
                if slot_idx == self._n_active - 1 and all(self._success_per_pad):
                    self.success_type = 'slip_success'

            # ── (7) Release ───────────────────────────────────────────────────
            self.move(self.atom.open_gripper(
                force=self.cfg.force_fast,
                steps=40,
            ))
            self.delay(10)

            # Re-check after release to capture post-slip resting position
            if self.check_pad_success(slot_idx):
                self._success_per_pad[slot_idx] = True

            print(
                f"[weight_sorting] slot {slot_idx} done | "
                f"weight='{weight_name}' | "
                f"placed_ok={self._success_per_pad[slot_idx]}"
            )

        # ── Final global success ───────────────────────────────────────────────
        if self.check_success() and self.success_type != 'slip_success':
            self.success_type = 'normal_success'