from ._base_task import *
import os
import numpy as np

OSCAR_BASE_POSE = Pose([0.55, 0.0, 0.1], [1, 0, 0, 0])


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    use_force_grasp: bool = True
    grasp_force: float = float(
        os.environ.get("LIFT_OSCAR_GRASP_FORCE", 10.0)
    )  # Newtons
    DENSITY: int = int(os.environ.get("LIFT_OSCAR_DENSITY", 500))

class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        print("////////// DENSITY //////////", self.cfg.DENSITY)
        print("////////// grasp_force_default //////////", self.cfg.grasp_force)
        oscar_pose = OSCAR_BASE_POSE.add_rotation([np.pi / 2, 0, 0])
        self.oscar = self._actor_manager.add_from_usd_file(
            name="oscar",
            asset_path="oscar.usd",
            pose=oscar_pose,
            density=self.cfg.DENSITY,
        )
        stand_pose = Pose([0.4, -0.08, 0.01], [1, 0, 0, 0])
        self.stand = self._actor_manager.add_from_usd_file(
            name="stand",
            asset_path="OrangePad.usd",
            pose=stand_pose,
        )

    def _reset_actors(self):
        oscar_offset = self.create_noise([0.01, 0.05, 0.0], [0, 0, np.pi / 18])
        oscar_pose = OSCAR_BASE_POSE.add_rotation([np.pi / 2, 0, 0]).add_offset(
            oscar_offset
        )
        self.oscar.set_pose(oscar_pose)
        self._success_steps = 0
        self._oscar_initial_pos = oscar_pose.p.copy()
        self.domain_rand_params = {
            "oscar_dx": float(oscar_offset.p[0]),
            "oscar_dy": float(oscar_offset.p[1]),
            "oscar_drz": float(oscar_offset.euler[2]),
            "density": self.cfg.DENSITY,
        }

    def pre_move(self):
        self.delay(10)
        oscar_pose = self.oscar.get_pose()
        # Top-down grasp at the oscar's center.
        # grasp_from=[0,0,1] → gripper approaches straight down, pre-position above the statue.
        # camera_up=[0,1,0] → fingers spread in world-y, gripping the widest axis of the body.
        target_pose = oscar_pose.add_bias([0, 0, 0.04], coord="world")
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0, 0, 1]),  # top-down approach (no table collision)
            np.array([1, 0, 0]),  # camera_up rotated 90° around z vs [0,1,0]
        )
        grasp_idx = self.oscar.register_point(pose=grasp_pose, type="contact")
        self.move(
            self.atom.grasp_actor(
                self.oscar,
                contact_point_id=grasp_idx,
                is_close=False,
            )
        )
        self.move(self.atom.close_gripper(0.0))

    def _print_effort(self, label: str):
        effort = self._robot_manager.get_gripper_effort()
        print(
            f"[{label}]  left={effort[0].item():+.2f} N  right={effort[1].item():+.2f} N"
        )

    def _play_once(self):
        self.move(self.atom.close_gripper(force=50, steps=50))
        self._print_effort("after default close")

        self.origin_inhand_pose = self.oscar.get_pose().rebase(self.atom.get_arm_pose())

        pad_pose = self.stand.get_pose()
        ee_now = self.atom.get_arm_pose()
        above_pad = Pose([pad_pose.p[0], pad_pose.p[1], ee_now.p[2] + 0.1], ee_now.q)
        self.move(self.atom.move_to_pose(above_pad), time_dilation_factor=0.5)
        self._print_effort("after lift")

        self.delay(30, is_save=True)

        print("[_play_once] opening gripper to release...")
        self.move(self.atom.open_gripper(force=50, steps=50))
        self._print_effort("after open")

        self.move(self.atom.move_by_displacement(z=-0.05), time_dilation_factor=0.5)
        # Final release onto the pad
        self.move(self.atom.open_gripper())

    def check_success(self):
        pad_pose = self.stand.get_pose()
        oscar_pos = self.oscar.get_pose().p
        in_zone = pad_pose.p[2] + 0.08 > oscar_pos[2] > pad_pose.p[2] + 0.03 and np.all(
            np.abs(oscar_pos[:2] - pad_pose.p[:2]) < 0.05
        )
        effort = (
            self._robot_manager.get_gripper_effort()
        )  # [left, right] in N; negative = closing
        print(
            f"oscar_z_offset={oscar_pos[2] - pad_pose.p[2]:.4f}  "
            f"effort  left={effort[0].item():+.2f} N  right={effort[1].item():+.2f} N"
        )
        self._success_steps = (self._success_steps + 1) if in_zone else 0
        return in_zone
