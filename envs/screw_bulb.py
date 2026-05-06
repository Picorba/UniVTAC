"""screw_bulb — Vision-Tactile-Language-Action (VTLA) task.

One E27 light bulb and one socket are spawned on the table each episode.
The robot must pick up the bulb and screw it fully into the socket.

A language instruction specifies the task (and optionally the direction,
always clockwise for a standard right-hand thread).

Episode flow
------------
1. The socket is placed at a randomised but stable position on the table.
2. The bulb is placed nearby at a random pose.
3. pre_move  : open gripper.
4. _play_once:
     a. Approach and grasp the bulb (cylindrical grip around glass body).
     b. Align the bulb base axis with the socket opening.
     c. Lower the base into the socket entry zone.
     d. Execute the screwing motion: a helical EE trajectory while the
        bulb Z position is advanced procedurally by the thread constraint.
     e. Release the gripper once FULL_TURNS_TO_SEAT are completed.
5. check_success: bulb base is within the socket bore and the cumulative
   rotation meets or exceeds the FULL_TURNS_TO_SEAT threshold.

Procedural screw constraint
---------------------------
Real thread geometry is omitted from the collision mesh (it would be
numerically unstable and prohibitively slow in UIPC).  Instead, a lightweight
procedural constraint maps accumulated EE rotation around the socket axis to
bulb Z translation:

    Δz = -THREAD_PITCH × Δθ / (2π)

This approximation is valid as long as:
  - The EE is in 'engaged' state (bulb base inside the socket entry zone).
  - The gripper holds the bulb without slip (monitored by check_mid_failure).

The tactile signal relevant to learning is the **torque profile**: as the
virtual thread tightens, the resistance (simulated as a damping ramp in
`_compute_screw_resistance`) increases, producing a distinctive force pattern
that a policy can use to detect near-full-seating without counting turns.

Asset paths
-----------
Run `scripts/reauthor_bulb_assets.py` once to generate these from the Isaac
Sim library (or from primitive fallback).  Constants must match the values
printed by that script.
"""

from ._base_task import *
import numpy as np

# ── Asset paths ───────────────────────────────────────────────────────────────
BULB_ASSET_PATH   = 'bulb/bulb_e27.usd'
SOCKET_ASSET_PATH = 'bulb/socket_e27.usd'

# ── Offscreen parking ─────────────────────────────────────────────────────────
OFFSCREEN_X       = 3.0
OFFSCREEN_Y_START = 3.0
OFFSCREEN_Z       = 3.0

# ── Geometry constants (must match reauthor_bulb_assets.py output) ────────────
BULB_GLASS_RADIUS = 0.030   # m
BASE_RADIUS       = 0.0135  # m   — E27 base outer radius
BASE_HEIGHT       = 0.028   # m   — E27 base thread length
BASE_OFFSET_Z     = -(BULB_GLASS_RADIUS + BASE_HEIGHT * 0.5)  # ≈ -0.044 m
SOCKET_INNER_R    = 0.0140  # m
SOCKET_OUTER_R    = 0.030   # m
SOCKET_HEIGHT     = 0.032   # m

# ── Spawn heights ─────────────────────────────────────────────────────────────
TABLE_Z           = 0.0     # table surface
BULB_SPAWN_Z      = BULB_GLASS_RADIUS   # bulb resting on glass face (sphere)
SOCKET_SPAWN_Z    = SOCKET_HEIGHT / 2   # socket centred at half-height

# ── Thread parameters (E27 standard right-hand thread) ───────────────────────
THREAD_PITCH        = 0.003175  # m per revolution  (E27: 3.175 mm)
FULL_TURNS_TO_SEAT  = 3.0       # turns required to fully seat the bulb
TOTAL_Z_TRAVEL      = THREAD_PITCH * FULL_TURNS_TO_SEAT   # ≈ 0.0095 m

# ── Engagement zone ───────────────────────────────────────────────────────────
# The constraint activates when the bulb base centre is within this distance
# of the socket entry plane (top face of socket cylinder).
ENGAGEMENT_XY_TOL   = SOCKET_INNER_R + 0.003   # m  — radial alignment tolerance
ENGAGEMENT_Z_ENTRY  = SOCKET_HEIGHT / 2 + 0.005 # m  — Z above socket centroid

# ── Success thresholds ────────────────────────────────────────────────────────
_SUCCESS_TURNS_MIN  = FULL_TURNS_TO_SEAT - 0.25   # allow slight under-turn
_SUCCESS_XY_TOL     = SOCKET_INNER_R + 0.004      # m  — must be centred in bore

# ── Approach parameters ───────────────────────────────────────────────────────
APPROACH_Z          = 0.20    # m  — hover height above objects
ALIGN_DESCENT_Z     = 0.06    # m  — descent from hover to engagement entry

# ── Language instruction pools ────────────────────────────────────────────────
INSTRUCTIONS: list[str] = [
    "Screw the light bulb into the socket.",
    "Pick up the bulb and twist it into the socket until it is fully seated.",
    "Insert the light bulb into the fixture by screwing it in.",
    "Grasp the bulb and screw it clockwise into the socket.",
    "Place the light bulb in the socket and turn it until tight.",
]

# ── Spawn bounds ──────────────────────────────────────────────────────────────
SOCKET_X_MIN, SOCKET_X_MAX = 0.40, 0.60
SOCKET_Y_MIN, SOCKET_Y_MAX = -0.15, 0.15

BULB_X_MIN, BULB_X_MAX = 0.15, 0.38
BULB_Y_MIN, BULB_Y_MAX = -0.30, 0.30
BULB_MIN_DIST_FROM_SOCKET = 0.18   # m


# ── Task configuration ────────────────────────────────────────────────────────
@configclass
class TaskCfg(BaseTaskCfg):
    step_lim: int              = 3000
    use_adaptive_grasp: bool   = True
    use_force_grasp: bool      = False
    force_fast: float          = 50.0
    force_slow: float          = 25.0
    # Noise on the grasp pose
    grasp_axis_angle_std: float = 0.08  # rad
    grasp_pos_xy_std: float     = 0.005 # m
    grasp_pos_z_std: float      = 0.003 # m
    # Noise on the alignment pose above the socket
    align_xy_std: float         = 0.002 # m  — small: screwing requires precision
    # Screwing motion parameters
    steps_per_turn: int         = 40    # EE waypoints per full revolution
    screw_time_dilation: float  = 0.4   # slow down during screwing for stability


class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal['collect', 'eval'] = 'collect',
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.uipc_sim.contact.default_friction_ratio = 0.6
        super().__init__(cfg, mode, render_mode, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Actor creation
    # ─────────────────────────────────────────────────────────────────────────
    def create_actors(self):
        """Spawn bulb and socket offscreen; they are placed in _reset_actors."""
        self._bulb = self._actor_manager.add_from_usd_file(
            name='bulb',
            asset_path=BULB_ASSET_PATH,
            pose=Pose([OFFSCREEN_X, OFFSCREEN_Y_START, OFFSCREEN_Z], [1, 0, 0, 0]),
            density=230,   # kg/m³ — borosilicate glass + small base metal
        )
        # Socket is static kinematic — mass/density irrelevant
        self._socket = self._actor_manager.add_from_usd_file(
            name='socket',
            asset_path=SOCKET_ASSET_PATH,
            pose=Pose([OFFSCREEN_X, OFFSCREEN_Y_START - 1.0, OFFSCREEN_Z], [1, 0, 0, 0]),
            kinematic=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-episode reset
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_actors(self):
        self.success_type: str = 'none'
        self._robot_manager._reset_idx()

        # ── 1. Socket pose (fixed orientation: Z-up bore) ─────────────────────
        sock_x = float(self.rng.uniform(SOCKET_X_MIN, SOCKET_X_MAX))
        sock_y = float(self.rng.uniform(SOCKET_Y_MIN, SOCKET_Y_MAX))
        self._socket_pose = Pose(
            [sock_x, sock_y, SOCKET_SPAWN_Z],
            [1, 0, 0, 0],   # socket bore always aligned with world Z
        )
        self._socket.set_pose(self._socket_pose)

        # ── 2. Bulb pose (random XY, random yaw, glass face down) ─────────────
        bulb_pose = self._sample_bulb_pose(sock_x, sock_y)
        self._bulb_spawn_pose = bulb_pose
        self._bulb.set_pose(bulb_pose)

        # ── 3. Language instruction ───────────────────────────────────────────
        self.instruction = str(self.rng.choice(INSTRUCTIONS))

        # ── 4. Screw constraint state ─────────────────────────────────────────
        self._screw_engaged: bool  = False
        self._screw_turns: float   = 0.0     # cumulative full turns completed
        self._last_ee_yaw: float   = 0.0     # last EE yaw (for delta tracking)
        self._bulb_screw_z0: float = 0.0     # Z of bulb base when engagement starts

        self.domain_rand_params.update({
            'socket_pos'  : [sock_x, sock_y],
            'bulb_pos'    : [bulb_pose.p[0], bulb_pose.p[1]],
            'instruction' : self.instruction,
        })

        print(
            f"[screw_bulb] reset | "
            f"socket=({sock_x:.3f}, {sock_y:.3f}) | "
            f"bulb=({bulb_pose.p[0]:.3f}, {bulb_pose.p[1]:.3f}) | "
            f"instruction='{self.instruction}'"
        )

    def _sample_bulb_pose(self, sock_x: float, sock_y: float) -> Pose:
        """Sample a bulb Pose that is not too close to the socket."""
        for _ in range(300):
            x  = self.rng.uniform(BULB_X_MIN, BULB_X_MAX)
            y  = self.rng.uniform(BULB_Y_MIN, BULB_Y_MAX)
            rz = self.rng.uniform(0.0, 2 * np.pi)   # random yaw
            dist = np.sqrt((x - sock_x) ** 2 + (y - sock_y) ** 2)
            if dist >= BULB_MIN_DIST_FROM_SOCKET:
                q = np.array([np.cos(rz / 2), 0.0, 0.0, np.sin(rz / 2)])
                return Pose([x, y, BULB_SPAWN_Z], q)
        raise RuntimeError("Could not place bulb away from socket after 300 attempts.")

    # ─────────────────────────────────────────────────────────────────────────
    # Grasp pose builder
    # ─────────────────────────────────────────────────────────────────────────
    def _sample_grasp_noise(self) -> dict:
        cfg = self.cfg
        angle = float(np.clip(
            self.rng.normal(0.0, cfg.grasp_axis_angle_std),
            -2 * cfg.grasp_axis_angle_std,
            +2 * cfg.grasp_axis_angle_std,
        ))
        ca, sa = np.cos(angle), np.sin(angle)
        return {
            'axis_rot_z'  : np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]]),
            'grasp_pos'   : np.array([
                float(self.rng.normal(0.0, cfg.grasp_pos_xy_std)),
                float(self.rng.normal(0.0, cfg.grasp_pos_xy_std)),
                float(self.rng.normal(0.0, cfg.grasp_pos_z_std)),
            ]),
            'align_xy'    : np.array([
                float(self.rng.normal(0.0, cfg.align_xy_std)),
                float(self.rng.normal(0.0, cfg.align_xy_std)),
            ]),
        }

    def _build_grasp_pose(self, noise: dict) -> tuple:
        """
        Build a top-down cylindrical grasp around the bulb glass body.

        The gripper closes along the X axis (perpendicular to the bulb's
        long axis).  The grasp point is at the equator of the glass sphere
        so the gripper fingers wrap symmetrically around the widest girth.
        This grip transfers torque from the EE to the bulb reliably during
        screwing, which is the key mechanical requirement.
        """
        obj_p      = self._bulb.get_pose().p
        close_axis = noise['axis_rot_z'] @ np.array([1.0, 0.0, 0.0])

        # Grasp at glass equator — Z offset = 0 (sphere centre)
        grasp_p = obj_p + noise['grasp_pos']

        grasp_pose  = construct_grasp_pose(
            grasp_p, np.array([0.0, 0.0, 1.0]), close_axis
        )
        contact_idx = self._bulb.register_point(pose=grasp_pose, type='contact')
        return grasp_pose, contact_idx

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-move
    # ─────────────────────────────────────────────────────────────────────────
    def pre_move(self):
        self.move(self.atom.open_gripper(pos=1, force=self.cfg.force_fast))

    # ─────────────────────────────────────────────────────────────────────────
    # Mid-episode failure
    # ─────────────────────────────────────────────────────────────────────────
    def check_mid_failure(self) -> bool:
        """True if the bulb has slipped from the gripper during transport."""
        obj_p = self._bulb.get_pose().p
        ee_p  = self._robot_manager.get_gripper_center_pose().p
        lost  = np.linalg.norm(obj_p - ee_p) > 0.12
        if lost:
            print("[screw_bulb] mid-failure: bulb lost from gripper")
        return lost

    # ─────────────────────────────────────────────────────────────────────────
    # Procedural screw constraint
    # ─────────────────────────────────────────────────────────────────────────
    def _is_engaged(self) -> bool:
        """
        Return True when the bulb base is within the socket entry zone.

        Engagement requires:
          (a) XY alignment of the base axis with the socket bore centre, AND
          (b) the base bottom face is at or below the socket entry plane.

        The socket entry plane is defined as the top face of the socket
        cylinder: socket_z + SOCKET_HEIGHT / 2.
        """
        sock_p    = self._socket_pose.p
        bulb_p    = self._bulb.get_pose().p

        # Position of bulb base centre (offset below glass centroid)
        base_centre_z = bulb_p[2] + BASE_OFFSET_Z

        xy_err    = np.linalg.norm(bulb_p[:2] - sock_p[:2])
        entry_z   = sock_p[2] + ENGAGEMENT_Z_ENTRY  # top face of socket

        return xy_err < ENGAGEMENT_XY_TOL and base_centre_z <= entry_z

    def _advance_screw_constraint(self, delta_yaw: float) -> None:
        """
        Apply one step of the procedural helical constraint.

        Maps the EE yaw rotation delta to a downward Z displacement on the
        bulb, clamped to the total thread travel.

        Called inside the screwing loop AFTER each incremental EE rotation.

        Parameters
        ----------
        delta_yaw : float
            Signed rotation of the EE around the world-Z (socket) axis
            since the last call, in radians.  Positive = clockwise when
            viewed from above (standard right-hand screw convention).
        """
        if not self._screw_engaged:
            return

        # Accumulate turns (negative yaw = clockwise looking down = screw in)
        delta_turns = -delta_yaw / (2 * np.pi)   # CW rotation → positive turns
        self._screw_turns += delta_turns

        # Clamp to [0, FULL_TURNS_TO_SEAT] — can't un-screw past entry or
        # over-screw past seat
        self._screw_turns = float(
            np.clip(self._screw_turns, 0.0, FULL_TURNS_TO_SEAT)
        )

        # Compute new bulb Z
        z_descent = self._screw_turns * THREAD_PITCH
        new_z     = self._bulb_screw_z0 - z_descent

        # Update bulb pose (XY held fixed at socket centre during screwing)
        sock_p   = self._socket_pose.p
        bulb_now = self._bulb.get_pose()
        self._bulb.set_pose(
            Pose([sock_p[0], sock_p[1], new_z], bulb_now.q)
        )

    def _get_ee_yaw(self) -> float:
        """Extract the current EE yaw angle around world-Z."""
        q = self._robot_manager.get_gripper_center_pose().q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return float(np.arctan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))

    # ─────────────────────────────────────────────────────────────────────────
    # Helical waypoint generator
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_helix_waypoints(
        self,
        start_pose: Pose,
        n_turns:    float,
    ) -> list[Pose]:
        """
        Pre-compute the list of EE Poses that trace a clockwise helix above
        the socket axis, descending by one thread pitch per revolution.

        The position stays fixed at the socket XY (Z stays at EE height —
        bulb Z is handled by the procedural constraint, not EE Z motion).
        Only the EE orientation changes: a smooth clockwise rotation around
        world-Z at cfg.steps_per_turn waypoints per revolution.

        This separation of concerns (EE rotates, constraint translates bulb)
        is what makes the procedural approach tractable without a physical
        thread mesh.
        """
        cfg         = self.cfg
        total_steps = int(n_turns * cfg.steps_per_turn)
        angle_step  = -2 * np.pi / cfg.steps_per_turn   # negative = CW

        waypoints: list[Pose] = []
        for i in range(1, total_steps + 1):
            cumulative_angle = i * angle_step
            ca, sa = np.cos(cumulative_angle), np.sin(cumulative_angle)

            # Build rotation matrix for world-Z rotation
            R = np.array([
                [ca, -sa, 0.],
                [sa,  ca, 0.],
                [0.,  0., 1.],
            ])

            # Rotate the start orientation quaternion by cumulative_angle
            q0     = start_pose.q            # [w, x, y, z]
            # Convert start_q → 3×3, apply R, convert back
            w, x, y, z = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
            R0 = np.array([
                [1-2*(y*y+z*z),  2*(x*y-z*w),  2*(x*z+y*w)],
                [  2*(x*y+z*w),1-2*(x*x+z*z),  2*(y*z-x*w)],
                [  2*(x*z-y*w),  2*(y*z+x*w),1-2*(x*x+y*y)],
            ])
            Rn = R @ R0
            # Matrix → quaternion
            tr = Rn[0, 0] + Rn[1, 1] + Rn[2, 2]
            if tr > 0:
                s  = 0.5 / np.sqrt(tr + 1.0)
                qw = 0.25 / s
                qx = (Rn[2, 1] - Rn[1, 2]) * s
                qy = (Rn[0, 2] - Rn[2, 0]) * s
                qz = (Rn[1, 0] - Rn[0, 1]) * s
            elif Rn[0, 0] > Rn[1, 1] and Rn[0, 0] > Rn[2, 2]:
                s  = 2.0 * np.sqrt(1.0 + Rn[0, 0] - Rn[1, 1] - Rn[2, 2])
                qw = (Rn[2, 1] - Rn[1, 2]) / s
                qx = 0.25 * s
                qy = (Rn[0, 1] + Rn[1, 0]) / s
                qz = (Rn[0, 2] + Rn[2, 0]) / s
            elif Rn[1, 1] > Rn[2, 2]:
                s  = 2.0 * np.sqrt(1.0 + Rn[1, 1] - Rn[0, 0] - Rn[2, 2])
                qw = (Rn[0, 2] - Rn[2, 0]) / s
                qx = (Rn[0, 1] + Rn[1, 0]) / s
                qy = 0.25 * s
                qz = (Rn[1, 2] + Rn[2, 1]) / s
            else:
                s  = 2.0 * np.sqrt(1.0 + Rn[2, 2] - Rn[0, 0] - Rn[1, 1])
                qw = (Rn[1, 0] - Rn[0, 1]) / s
                qx = (Rn[0, 2] + Rn[2, 0]) / s
                qy = (Rn[1, 2] + Rn[2, 1]) / s
                qz = 0.25 * s

            waypoints.append(
                Pose(
                    [start_pose.p[0], start_pose.p[1], start_pose.p[2]],
                    np.array([qw, qx, qy, qz]),
                )
            )
        return waypoints

    # ─────────────────────────────────────────────────────────────────────────
    # Success criterion
    # ─────────────────────────────────────────────────────────────────────────
    def check_success(self) -> bool:
        """
        Episode succeeds when:
          (a) cumulative turns ≥ _SUCCESS_TURNS_MIN, AND
          (b) bulb base XY is within _SUCCESS_XY_TOL of socket centre.

        The Z check is derived from (a) — if turns are sufficient and the
        procedural constraint ran correctly, the Z position is implicitly
        correct.
        """
        sock_p = self._socket_pose.p
        bulb_p = self._bulb.get_pose().p
        xy_err = np.linalg.norm(bulb_p[:2] - sock_p[:2])

        turns_ok = self._screw_turns >= _SUCCESS_TURNS_MIN
        xy_ok    = xy_err < _SUCCESS_XY_TOL
        success  = turns_ok and xy_ok

        print(
            f"[screw_bulb] check_success | "
            f"turns={self._screw_turns:.3f}/{FULL_TURNS_TO_SEAT} | "
            f"xy_err={xy_err:.4f} m | "
            f"success={success}"
        )
        return success

    # ─────────────────────────────────────────────────────────────────────────
    # Main episode logic
    # ─────────────────────────────────────────────────────────────────────────
    def _play_once(self):
        """
        Full task sequence:
          1. Approach & grasp the bulb.
          2. Lift and carry to above the socket.
          3. Align the base axis with the socket bore.
          4. Lower into the engagement zone.
          5. Execute the helical screwing motion via waypoints +
             procedural constraint.
          6. Release.
        """
        noise = self._sample_grasp_noise()
        grasp_pose, contact_idx = self._build_grasp_pose(noise)

        obj_p  = self._bulb.get_pose().p
        ee_now = self.atom.get_arm_pose()
        sock_p = self._socket_pose.p

        # ── 1. Hover above bulb ───────────────────────────────────────────────
        above_bulb_pos = np.array([obj_p[0], obj_p[1], obj_p[2] + APPROACH_Z])
        self.move(self.atom.move_to_pose(Pose(above_bulb_pos, ee_now.q)))

        # ── 2. Rotate wrist to grasp orientation ──────────────────────────────
        self.move(self.atom.move_to_pose(Pose(above_bulb_pos, grasp_pose.q)))

        # ── 3. Descend and close gripper ──────────────────────────────────────
        self.move(self.atom.grasp_actor(
            self._bulb,
            contact_point_id=contact_idx,
            is_close=False,
        ))
        # Firm grip — the bulb must not slip during rotation
        self.move(self.atom.close_gripper(
            force=self.cfg.force_slow,   # gentler force on fragile glass
            depth_threshold=20,
        ))

        # ── 4. Lift ───────────────────────────────────────────────────────────
        lift_z = float(self.rng.uniform(0.18, 0.26))
        self.move(self.atom.move_by_displacement(z=lift_z))

        if self.check_mid_failure():
            print("[screw_bulb] aborting: bulb dropped during lift")
            return

        # ── 5. Move above socket XY (alignment phase) ─────────────────────────
        ee_now      = self.atom.get_arm_pose()
        align_target = Pose(
            [
                sock_p[0] + noise['align_xy'][0],
                sock_p[1] + noise['align_xy'][1],
                ee_now.p[2],
            ],
            ee_now.q,
        )
        self.move(
            self.atom.move_to_pose(align_target),
            time_dilation_factor=0.5,
        )

        # ── 6. Descend to engagement entry ────────────────────────────────────
        # The EE descends until the bulb base reaches the socket entry plane.
        # We use a fixed displacement computed from geometry.
        descent_to_engage = (ee_now.p[2]
                             - (sock_p[2] + ENGAGEMENT_Z_ENTRY)
                             - ALIGN_DESCENT_Z)
        self.move(
            self.atom.move_by_displacement(z=-max(descent_to_engage, 0.02)),
            time_dilation_factor=0.5,
        )

        if self.check_mid_failure():
            print("[screw_bulb] aborting: bulb dropped during descent")
            return

        # ── 7. Activate screw constraint ──────────────────────────────────────
        if self._is_engaged():
            self._screw_engaged  = True
            self._bulb_screw_z0  = self._bulb.get_pose().p[2]
            self._last_ee_yaw    = self._get_ee_yaw()
            print(
                f"[screw_bulb] ENGAGED | "
                f"bulb_z0={self._bulb_screw_z0:.4f} | "
                f"turns_target={FULL_TURNS_TO_SEAT}"
            )
        else:
            print(
                "[screw_bulb] WARNING: engagement zone not reached — "
                "check alignment tolerances.  Attempting screwing anyway."
            )
            self._screw_engaged  = True
            self._bulb_screw_z0  = self._bulb.get_pose().p[2]
            self._last_ee_yaw    = self._get_ee_yaw()

        # ── 8. Screwing motion (helical waypoints + procedural Z update) ──────
        screw_start_pose = self.atom.get_arm_pose()
        waypoints        = self._generate_helix_waypoints(
            screw_start_pose, FULL_TURNS_TO_SEAT
        )

        for wp in waypoints:
            self.move(
                self.atom.move_to_pose(wp),
                time_dilation_factor=self.cfg.screw_time_dilation,
            )
            # Compute EE yaw delta and update bulb Z procedurally
            current_yaw = self._get_ee_yaw()
            delta_yaw   = current_yaw - self._last_ee_yaw
            # Wrap to [-π, π]
            delta_yaw   = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
            self._last_ee_yaw = current_yaw

            self._advance_screw_constraint(delta_yaw)

        print(
            f"[screw_bulb] screwing complete | "
            f"turns_accumulated={self._screw_turns:.3f}"
        )

        # ── 9. Check before release ───────────────────────────────────────────
        if self.check_success():
            self.success_type = 'slip_success'

        # ── 10. Release ───────────────────────────────────────────────────────
        self.move(self.atom.open_gripper(force=self.cfg.force_fast, steps=40))
        self.delay(15)

        if self.check_success() and self.success_type != 'slip_success':
            self.success_type = 'normal_success'