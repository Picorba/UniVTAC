from ._base_task import *
import numpy as np

# N visually identical cuboids (Bar_Cuboid.usd, AffineBody) with different densities.
# Phase 1 – weigh every cube: grasp → lift → hold → read gripper effort → return.
# Phase 2 – place in ascending mass order into numbered bins (pad 0 = lightest).

_N_DEFAULT = 4

# Densities span a 15× range so effort differences are clearly tactile.
_DENSITY_MIN =  100.0   # kg/m³  (lightest)
_DENSITY_MAX = 1500.0   # kg/m³  (heaviest)

# Bar_Cuboid geometry (from USD): 3 cm × 1.5 cm × 12 cm, local origin at bottom.
# Grasp wings protrude in Y (±1.5 cm) at local z = 0.095–0.105, centred at 0.10.
_CUBE_SPAWN_Z  = 0.001  # place bar bottom just above the table surface

# Starting row: cubes lined up along Y at fixed X
_START_X       = 0.55
_START_Z       = _CUBE_SPAWN_Z
_START_SPACING = 0.13    # centre-to-centre gap (m)

# Destination pads: parallel row, shifted toward the robot
_PAD_X       = 0.40
_PAD_SPACING = 0.13
_PAD_Z       = 0.005

# Grasp motion
# _APPROACH_Z: height offset from bar bottom to hover at before descending (> 0.12 to clear the top)
# _GRASP_Z:    height offset from bar bottom to the wing centre (= 0.10)
_APPROACH_Z   = 0.18   # world z = bar_bottom + 0.18 → 6 cm above bar top
_GRASP_Z      = 0.09   # world z = bar_bottom + 0.09 → bottom edge of grasp wings (0.095–0.105)

# Weighing motion
_WEIGH_LIFT_Z = 0.18
_WEIGH_HOLD   = 25   # sim steps held at lift height before reading effort

# Clearance lift before any horizontal transport (must clear bar top at 0.12 m + neighbours)
_TRANSPORT_LIFT_Z = 0.20   # lift from grasp point before moving to pad


def _centred_y(n: int, spacing: float) -> list[float]:
    """Return n y-positions centred on 0 with the given spacing."""
    half = (n - 1) / 2.0
    return [spacing * (i - half) for i in range(n)]


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim: int    = 4000
    n_cubes:  int    = _N_DEFAULT
    use_force_grasp:     bool  = True
    grasp_force:         float = 12.0   # gentle force when closing onto the object
    fast_gripper_force:  float = 50.0   # high force for fast open / fast release


class Task(BaseTask):

    def create_actors(self):
        n = self.cfg.n_cubes
        # Canonical density ranks: index 0 = lightest, index n-1 = heaviest
        self._densities: np.ndarray = np.linspace(_DENSITY_MIN, _DENSITY_MAX, n)

        start_ys = _centred_y(n, _START_SPACING)
        self._cubes: list[Actor] = []
        for i, density in enumerate(self._densities):
            cube = self._actor_manager.add_from_usd_file(
                name=f'cube_{i}',
                asset_path='Bar_Cuboid.usd',
                pose=Pose([_START_X, start_ys[i], _START_Z], [1, 0, 0, 0]),
                density=float(density),
            )
            self._cubes.append(cube)

        # Destination pads – fixed positions, pad index == weight rank
        pad_ys = _centred_y(n, _PAD_SPACING)
        self._pads: list[Actor] = []
        for i in range(n):
            pad = self._actor_manager.add_from_usd_file(
                name=f'pad_{i}',
                asset_path='GreenPad.usd',
                pose=Pose([_PAD_X, pad_ys[i], _PAD_Z], [1, 0, 0, 0]),
                density=1e5,
            )
            self._pads.append(pad)

    def _reset_actors(self):
        n = self.cfg.n_cubes
        # Assign a random permutation of density indices to starting slots
        perm = self.rng.permutation(n)   # perm[slot] = density_index
        self._density_perm = perm

        start_ys = _centred_y(n, _START_SPACING)
        for slot, density_idx in enumerate(perm):
            noise = self.create_noise([0.005, 0.005, 0.0], [0, 0, np.pi / 18])
            pose  = Pose([_START_X, start_ys[slot], _START_Z], [1, 0, 0, 0]).add_offset(noise)
            self._cubes[slot].set_pose(pose)

        self._weighed_efforts: list[float | None] = [None] * n
        self._placed_order:    list[int]           = []
        self._success_steps:   int                 = 0

    # ── Episode logic ─────────────────────────────────────────────────────────

    def _top_down_grasp(self, cube: Actor):
        """Open gripper, approach top-down, descend to wing height, close.

        Uses grasp_actor with pre_dis so cuRobo plans the full above→descend
        motion as a single two-waypoint problem — avoids the S-curve artefact
        that appears when move_to_pose/move_by_displacement are chained separately.

        Contact point is registered at the wing centre (local z = _GRASP_Z) with
        top-down orientation, so get_grasp_pose raises it by pre_dis to the
        approach height automatically.
        """
        # Place contact point at wing height with top-down orientation
        cube_p = cube.get_pose().p
        grasp_p = cube_p + np.array([0, 0, _GRASP_Z])
        contact_pose = construct_grasp_pose(
            grasp_p, np.array([0, 0, 1]), np.array([1, 0, 0])
        )
        cube.cfg.contact_points.clear()
        contact_idx = cube.register_point(pose=contact_pose, type='contact')

        # pre_dis lifts the pre-grasp waypoint to _APPROACH_Z above the bar bottom
        pre_dis = _APPROACH_Z - _GRASP_Z   # 0.18 - 0.09 = 0.09 m


        # grasp_actor passes pre_dis to plan_arm → cuRobo sees both waypoints and
        # plans a clean straight descent rather than a freeform arc
        self.move(self.atom.grasp_actor(
            cube,
            contact_point_id=contact_idx,
            pre_dis=pre_dis,
            dis=0,
            is_close=False,
        ))

        # Close to pos=0.0 with explicit force; stops when position reached or stable
        self.move(self.atom.close_gripper(pos=0.0, force=self.cfg.grasp_force))

    def pre_move(self):
        self.delay(1)

    def _play_once(self):
        n = self.cfg.n_cubes

        # ── Phase 1: weigh each cube ──────────────────────────────────────────
        print(f'[weight_sorting] Phase 1 – weighing {n} cubes')
        for slot in range(n):
            cube = self._cubes[slot]
            self._top_down_grasp(cube)

            self.move(self.atom.move_by_displacement(z=_WEIGH_LIFT_Z))
            self.delay(_WEIGH_HOLD, is_save=True)

            effort = self._robot_manager.get_gripper_effort()
            proxy  = (abs(effort[0].item()) + abs(effort[1].item())) / 2.0
            self._weighed_efforts[slot] = proxy

            true_density = self._densities[self._density_perm[slot]]
            print(f'  slot {slot}: effort_proxy={proxy:.3f} N  '
                  f'density={true_density:.0f} kg/m³')

            self.move(self.atom.move_by_displacement(z=-_WEIGH_LIFT_Z))
            self.move(self.atom.open_gripper(pos=1.0, force=self.cfg.fast_gripper_force))
            self.delay(20)

        # ── Phase 2: place light → heavy ──────────────────────────────────────
        print('[weight_sorting] Phase 2 – sorting')
        ranked_slots = sorted(range(n), key=lambda s: self._weighed_efforts[s])
        for dest_idx, slot in enumerate(ranked_slots):
            cube = self._cubes[slot]
            self._top_down_grasp(cube)

            # Lift clear of all neighbouring bars before moving horizontally
            self.move(self.atom.move_by_displacement(z=_TRANSPORT_LIFT_Z))
            self.move(self.atom.place_actor(cube, self._pads[dest_idx].get_pose(), is_open=True))
            self._placed_order.append(slot)
            print(f'  slot {slot} → pad {dest_idx}')

    # ── Success criterion ──────────────────────────────────────────────────────

    def check_success(self) -> bool:
        n = self.cfg.n_cubes
        if len(self._placed_order) < n:
            return False
        # Ground-truth ascending rank sorted by actual density
        true_rank = sorted(range(n), key=lambda s: self._densities[self._density_perm[s]])
        correct   = sum(p == t for p, t in zip(self._placed_order, true_rank))
        rank_acc  = correct / n
        success   = rank_acc >= 1.0
        print(f'[weight_sorting] rank_accuracy={rank_acc:.2f}  success={success}')
        self._success_steps = (self._success_steps + 1) if success else 0
        return success
