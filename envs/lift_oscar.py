"""lift_oscar — lift task with a contact force sensor on the manipulated object.

A cylinder is spawned as a PhysX RigidActor (activate_contact_sensors=True) rather than
a UIPC Actor.  This enables Isaac Lab's ContactSensor to report the net contact force
acting on the object each step.

Trade-off: the UIPC gelpad tactile contact is disabled for the cylinder (it only works with
UIPC bodies), but PhysX correctly simulates the gripper–object contact forces.
"""

from ._base_task import *
from .sensors.contact import ObjectContactSensorCfg
import os
import numpy as np

CYLINDER_BASE_POSE = Pose([0.55, 0.0, 0.03], [1, 0, 0, 0])


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    use_adaptive_grasp = False
    use_force_grasp: bool = True
    grasp_force: float = float(os.environ.get("LIFT_OSCAR_GRASP_FORCE", 10.0))
    DENSITY: int = int(os.environ.get("LIFT_OSCAR_DENSITY", 500))

    contact_sensors: list = [
        ObjectContactSensorCfg(
            name='cylinder',
            prim_path='/World/envs/env_.*/cylinder',
        )
    ]


class Task(BaseTask):
    def create_actors(self):
        # RigidActor with contact sensing enabled — no UIPC FEM, but PhysX ContactSensor works.
        self.cylinder = self._actor_manager.add_rigid_from_usd_file(
            name='cylinder',
            asset_path='oscar.usd',
            pose=CYLINDER_BASE_POSE,
            density=self.cfg.DENSITY,
            activate_contact_sensors=True,
        )
        stand_pose = Pose([0.4, -0.08, 0.01], [1, 0, 0, 0])
        self.stand = self._actor_manager.add_rigid_from_usd_file(
            name='stand',
            asset_path='OrangePad.usd',
            pose=stand_pose,
        )

    def _reset_actors(self):
        cylinder_offset = self.create_noise([0.01, 0.05, 0.0], [0, 0, np.pi / 18])
        cylinder_pose = CYLINDER_BASE_POSE.add_offset(cylinder_offset)
        self.cylinder.set_pose(cylinder_pose)
        self._success_steps = 0
        self._cylinder_initial_pos = cylinder_pose.p.copy()
        self.domain_rand_params = {
            'cylinder_dx': float(cylinder_offset.p[0]),
            'cylinder_dy': float(cylinder_offset.p[1]),
            'cylinder_drz': float(cylinder_offset.euler[2]),
            'density': self.cfg.DENSITY,
        }

    def pre_move(self):
        self.delay(10)
        cylinder_pose = self.cylinder.get_pose()
        target_pose = cylinder_pose.add_bias([0, 0, 0.04], coord='world')
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
        )
        grasp_idx = self.cylinder.register_point(pose=grasp_pose, type='contact')
        self.move(self.atom.grasp_actor(self.cylinder, contact_point_id=grasp_idx, is_close=False))
        self.move(self.atom.close_gripper(0.0))

    def _print_forces(self, label: str):
        gripper_effort = self._robot_manager.get_gripper_effort()
        contact_obs = self._contact_sensor_manager.get_observations()
        cylinder_force = contact_obs['cylinder']['net_force']
        cylinder_mag = contact_obs['cylinder']['magnitude']
        print(
            f'[{label}]'
            f'  gripper effort  L={gripper_effort[0].item():+.2f} N  R={gripper_effort[1].item():+.2f} N'
            f'  |  cylinder contact force  {cylinder_force.tolist()}  |F|={cylinder_mag.item():.2f} N'
        )

    def _play_once(self):
        self.move(self.atom.close_gripper(force=self.cfg.grasp_force, steps=50))
        self._print_forces('after grasp')

        self.origin_inhand_pose = self.cylinder.get_pose().rebase(self.atom.get_arm_pose())

        pad_pose = self.stand.get_pose()
        ee_now = self.atom.get_arm_pose()
        above_pad = Pose([pad_pose.p[0], pad_pose.p[1], ee_now.p[2] + 0.1], ee_now.q)
        self.move(self.atom.move_to_pose(above_pad), time_dilation_factor=0.5)
        self._print_forces('after lift')

        self.delay(30, is_save=True)

        self.move(self.atom.open_gripper(force=self.cfg.grasp_force, steps=50))
        self._print_forces('after release')

        self.move(self.atom.move_by_displacement(z=-0.05), time_dilation_factor=0.5)
        self.move(self.atom.open_gripper())

    def check_success(self):
        pad_pose = self.stand.get_pose()
        cylinder_pos = self.cylinder.get_pose().p
        in_zone = (
            pad_pose.p[2] + 0.08 > cylinder_pos[2] > pad_pose.p[2] + 0.03
            and np.all(np.abs(cylinder_pos[:2] - pad_pose.p[:2]) < 0.05)
        )
        contact_obs = self._contact_sensor_manager.get_observations()
        cylinder_mag = contact_obs['cylinder']['magnitude'].item()
        print(
            f'cylinder_z_offset={cylinder_pos[2] - pad_pose.p[2]:.4f}'
            f'  contact_force={cylinder_mag:.2f} N'
        )
        self._success_steps = (self._success_steps + 1) if in_zone else 0
        return in_zone
