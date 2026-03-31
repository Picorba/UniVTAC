from ._base_task import *
import numpy as np

# ── Hammer geometry constants (must match scripts/gen_hammer.py) ──────────────
HANDLE_L = 0.30    # handle length along local x-axis (m)
HANDLE_W = 0.015   # handle square cross-section side (m)

# Head dimensions per variant (depth × width × height, all in metres)
VARIANTS: dict[str, dict] = {
    'light':  dict(depth=0.03, width=0.05, height=0.03),
    'medium': dict(depth=0.04, width=0.07, height=0.04),
    'heavy':  dict(depth=0.05, width=0.09, height=0.05),
}

# Two density options: same CoM geometry, but different inertia / grip force
DENSITIES = [500, 2000]   # kg/m³  (light mass vs heavy mass)


def _com_x(depth: float, width: float, height: float) -> float:
    """CoM along local x-axis (from handle centre) for uniform-density hammer."""
    v_handle = HANDLE_L * HANDLE_W ** 2
    v_head   = depth * width * height
    return v_head * (HANDLE_L / 2 + depth / 2) / (v_handle + v_head)


# Pre-computed per-variant look-up tables
COM_X   = {n: _com_x(**p) for n, p in VARIANTS.items()}
# spawn_z: bottom of head flush with table (z=0) + 2 mm clearance
SPAWN_Z = {n: p['height'] / 2 + 0.002 for n, p in VARIANTS.items()}

# ── Data-collection policy settings ──────────────────────────────────────────
# fraction of demos that include a deliberate off-centre probe step
SEARCH_PROB       = 0.7
# probe offset sampled uniformly from [MIN, MAX] on each side of CoM (m)
MAX_SEARCH_OFFSET = 0.07
MIN_SEARCH_OFFSET = 0.025
# normalised gripper joint target for a 15 mm wide handle
GRIPPER_NORM_POS  = 0.008 / 0.039


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(1, 0.0, 0.15), rot=(0.5, 0.5, 0.5, 0.5), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480, height=270, update_period=1/120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480, height=270, update_period=1/120,
        ),
    ]
    use_adaptive_grasp = False
    step_lim = 600


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect',
                 render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction  = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def create_actors(self):
        """Spawn all (variant × density) combinations off-screen.

        3 head sizes  ×  2 densities  =  6 actors total.
        One is moved into the workspace each episode.
        """
        self._hammers: dict[tuple, object] = {}
        idx = 0
        for variant, params in VARIANTS.items():
            for density in DENSITIES:
                key = (variant, density)
                self._hammers[key] = self._actor_manager.add_from_usd_file(
                    name=f'hammer_{variant}_{density}',
                    asset_path=f'Hammer_{variant}.usd',
                    pose=Pose([0.4, 1.0 + idx * 0.25, 0.01], [1, 0, 0, 0]),
                    density=density,
                )
                idx += 1

    # ── Per-episode reset ──────────────────────────────────────────────────────

    def _reset_actors(self):
        variant = self.rng.choice(list(VARIANTS.keys()))
        density = int(self.rng.choice(DENSITIES))
        self.hammer   = self._hammers[(variant, density)]
        self._com_x   = COM_X[variant]
        self._variant = variant

        # Clear accumulated contact-point registrations from previous episodes
        self.hammer.cfg.contact_points.clear()

        # Flat on table, handle pointing in a random horizontal direction
        theta      = self.rng.uniform(0, 2 * np.pi)
        spawn_pose = Pose(
            [0.38, 0.0, SPAWN_Z[variant]], [1, 0, 0, 0]
        ).add_rotation([0, 0, theta], coord='world')
        self.hammer.set_pose(spawn_pose)

        self._success_steps = 0
        self.domain_rand_params = {
            'variant': variant,
            'density': density,
            'com_x':   float(self._com_x),
            'theta':   float(theta),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_hammer_state(self):
        """Read current hammer pose and refresh all cached handle/CoM fields."""
        pose    = self.hammer.get_pose()
        dir_3d  = pose.R @ np.array([1.0, 0.0, 0.0])
        dir_xy  = np.array([dir_3d[0], dir_3d[1], 0.0])
        dir_xy /= np.linalg.norm(dir_xy)

        self._hammer_origin  = pose.p.copy()
        self._handle_dir_3d  = dir_3d
        self._handle_dir_xy  = dir_xy
        # EE height for a top-down grasp: 4 cm above the handle centre
        self._handle_grasp_z = pose.p[2] + 0.04
        self._com_world      = pose.p + dir_3d * self._com_x

    def _grasp_handle_at(self, world_pos: np.ndarray):
        """Plan and execute a top-down grasp at world_pos (x, y), fixed EE height."""
        grasp_p = np.array([world_pos[0], world_pos[1], self._handle_grasp_z])
        # camera_up = handle direction → fingers straddle the handle perpendicularly
        cpose = construct_grasp_pose(grasp_p, [0, 0, 1], self._handle_dir_xy)
        cid   = self.hammer.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.hammer, contact_point_id=cid, pre_dis=0.05, dis=0.0, is_close=False
        ))
        self.move(self.atom.close_gripper(GRIPPER_NORM_POS))

    # ── Episode logic ─────────────────────────────────────────────────────────

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.5))
        # Cache hammer state; _play_once will also refresh it after each release
        self._update_hammer_state()

    def _play_once(self):
        # ── Phase 1: deliberate off-CoM probe (SEARCH_PROB of all demos) ─────
        # These frames are saved and teach the policy to associate tilt with
        # the need to shift its grip toward the heavier side.
        if self.rng.random() < SEARCH_PROB:
            sign      = self.rng.choice([-1.0, 1.0])
            magnitude = self.rng.uniform(MIN_SEARCH_OFFSET, MAX_SEARCH_OFFSET)

            # Clamp so the probe point stays within ±12 cm of handle centre
            probe_local_x = float(np.clip(
                self._com_x + sign * magnitude, -0.12, 0.12
            ))
            probe_world = self._hammer_origin + self._handle_dir_3d * probe_local_x

            self._grasp_handle_at(probe_world)

            # Lift slightly – the resulting tilt is the key learning signal
            self.move(self.atom.move_by_displacement(z=0.05), time_dilation_factor=0.5)
            self.delay(20)

            # Lower and release; give the hammer time to settle
            self.move(self.atom.move_by_displacement(z=-0.05), time_dilation_factor=0.5)
            self.move(self.atom.open_gripper(0.5))
            self.delay(25)

            # Refresh hammer state (it may have shifted/rotated when released)
            self._update_hammer_state()
            self.hammer.cfg.contact_points.clear()

        # ── Phase 2: definitive grasp at the analytically known CoM ──────────
        self._grasp_handle_at(self._com_world)
        self.move(self.atom.move_by_displacement(z=0.20), time_dilation_factor=0.5)
        self.delay(30)

    # ── Termination conditions ────────────────────────────────────────────────

    def check_success(self):
        pose      = self.hammer.get_pose()
        long_axis = pose.to_transformation_matrix()[:3, 0]   # local x in world
        # |sin(tilt from horizontal)| < sin(20°) ≈ 0.342
        tilt_sin  = float(abs(np.dot(long_axis, [0.0, 0.0, 1.0])))
        is_high       = pose.p[2] > 0.15
        is_horizontal = tilt_sin < np.sin(np.deg2rad(20))

        self._success_steps = (self._success_steps + 1) if (is_high and is_horizontal) else 0
        return is_high and is_horizontal

    def check_early_stop(self):
        ee_pos     = self.atom.get_arm_pose().p
        hammer_pos = self.hammer.get_pose().p
        # Stop if the hammer has drifted far from the workspace centre
        # AND is no longer near the end-effector (i.e. it was dropped/knocked)
        far_from_workspace = np.linalg.norm(hammer_pos[:2] - np.array([0.38, 0.0])) > 0.35
        far_from_ee        = np.linalg.norm(hammer_pos - ee_pos) > 0.20
        return far_from_workspace and far_from_ee
