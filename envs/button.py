from ._base_task import *
import numpy as np

# Button A = left, Button B = right
BUTTON_A_BASE_POSE = Pose([0.50, -0.08, 0.025], [1, 0, 0, 0])
BUTTON_B_BASE_POSE = Pose([0.50,  0.08, 0.025], [1, 0, 0, 0])

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 600

    # How far above the button surface the fingertip probes (metres)
    probe_height_offset = 0.005

    # Force threshold (N) below which we consider "no resistance" → broken
    broken_force_threshold = 0.3

    # Position noise applied independently to each button each episode
    button_pos_noise_xy = 0.01   # ± metres in x/y
    button_rot_noise_z  = np.pi / 24  # ± ~7.5°


class Task(BaseTask):
    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def create_actors(self):
        self.button_a = self._actor_manager.add_from_usd_file(
            name='button_a',
            asset_path="button.usd",
            pose=BUTTON_A_BASE_POSE,
        )
        self.button_b = self._actor_manager.add_from_usd_file(
            name='button_b',
            asset_path="button.usd",   # identical visual asset
            pose=BUTTON_B_BASE_POSE,
        )
        # Flat surface the buttons sit on (optional, for visual grounding)
        self.table_pad = self._actor_manager.add_from_usd_file(
            name='table_pad',
            asset_path="OrangePad.usd",
            pose=Pose([0.50, 0.0, 0.01], [1, 0, 0, 0]),
        )

    # ------------------------------------------------------------------
    # Per-episode reset
    # ------------------------------------------------------------------
    def _reset_actors(self):
        # Randomly decide which button is broken this episode
        self.broken_button_idx = np.random.randint(0, 2)   # 0 = A, 1 = B

        # Independent positional noise for each button
        offset_a = self.create_noise(
            [self.cfg.button_pos_noise_xy, self.cfg.button_pos_noise_xy, 0.0],
            [0, 0, self.cfg.button_rot_noise_z]
        )
        offset_b = self.create_noise(
            [self.cfg.button_pos_noise_xy, self.cfg.button_pos_noise_xy, 0.0],
            [0, 0, self.cfg.button_rot_noise_z]
        )

        self.button_a.set_pose(BUTTON_A_BASE_POSE.add_offset(offset_a))
        self.button_b.set_pose(BUTTON_B_BASE_POSE.add_offset(offset_b))

        # Apply broken physics: lock the broken button's press joint
        broken, working = self._get_ordered_buttons()
        broken.set_joint_stiffness(1e6)   # effectively rigid
        broken.set_joint_damping(1e4)
        working.set_joint_stiffness(50.0)  # normal spring feel
        working.set_joint_damping(5.0)

        self.domain_rand_params = {
            'broken_button':   self.broken_button_idx,
            'button_a_dx':     float(offset_a.p[0]),
            'button_a_dy':     float(offset_a.p[1]),
            'button_a_drz':    float(offset_a.euler[2]),
            'button_b_dx':     float(offset_b.p[0]),
            'button_b_dy':     float(offset_b.p[1]),
            'button_b_drz':    float(offset_b.euler[2]),
        }

        # Will be filled during _play_once
        self.agent_guess = None
        self.probe_forces = {0: [], 1: []}

    # ------------------------------------------------------------------
    # Pre-move: position the gripper above the scene centre
    # ------------------------------------------------------------------
    def pre_move(self):
        self.delay(10)

        # Hover above the midpoint between the two buttons
        mid_pose = Pose(
            [0.50, 0.0, 0.15],   # high enough to plan to either button
            [1, 0, 0, 0]
        )
        self.move(self.atom.move_to_pose(mid_pose))

    # ------------------------------------------------------------------
    # Main task: probe both buttons, then press the working one
    # ------------------------------------------------------------------
    def _play_once(self):
        for btn_idx, button in enumerate([self.button_a, self.button_b]):
            self._probe_button(btn_idx, button)

        # Decide: broken button is the one that resisted (low displacement / high force spike)
        self.agent