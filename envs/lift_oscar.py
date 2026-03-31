from ._base_task import *
import os
import numpy as np


OSCAR_BASE_POSE = Pose([0.55, 0.0, 0.03], [1, 0, 0, 0])


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 7.5
    use_adaptive_grasp = False
    use_force_grasp: bool = True
    grasp_force: float = float(os.environ.get('LIFT_OSCAR_GRASP_FORCE', 40.0))  # Newtons
    DENSITY: int = int(os.environ.get('LIFT_OSCAR_DENSITY', 500))


class Task(BaseTask):
    def create_actors(self):
        oscar_pose = OSCAR_BASE_POSE.add_rotation([0, np.pi/2, np.pi/2])
        print("////////// DENSITY //////////", self.cfg.DENSITY)
        wall_pose = Pose([0.75, 0.0, 0.005], [1, 0, 0, 0])
        oscar_pose = wall_pose.add_bias([-0.18, 0.0, 0.03]).add_rotation([0, np.pi/2, np.pi/2])

        self.wall = self._actor_manager.add_from_usd_file(
            name='wall',
            asset_path="Wall.usd",
            pose=wall_pose,
            density=1e5
        )
        """self.oscar = self._actor_manager.add_rigid_from_usd_file(
            name='oscar',
            asset_path="oscar.usd",
            pose=oscar_pose
        )"""
        self.oscar = self._actor_manager.add_from_usd_file(
            name='oscar', asset_path="oscar.usd", pose=oscar_pose, density=self.cfg.DENSITY
        )
        stand_pose = Pose([0.4, -0.08, 0.01], [1, 0, 0, 0])
        self.stand = self._actor_manager.add_from_usd_file(
            name='stand', asset_path="OrangePad.usd", pose=stand_pose,
        )

    def _reset_actors(self):
        oscar_offset = self.create_noise([0.01, 0.05, 0.0], [0, 0, np.pi/18])
        oscar_pose = OSCAR_BASE_POSE.add_rotation([np.pi/2, 0, 0]).add_offset(oscar_offset)
        self.oscar.set_pose(oscar_pose)
        self._success_steps = 0
        self._oscar_initial_pos = oscar_pose.p.copy()
        self.domain_rand_params = {
            'oscar_dx': float(oscar_offset.p[0]),
            'oscar_dy': float(oscar_offset.p[1]),
            'oscar_drz': float(oscar_offset.euler[2]),
            'density': self.cfg.DENSITY,
        }

    def pre_move(self):
        self.delay(10)
        oscar_pose = self.oscar.get_pose()
        # Top-down grasp at the oscar's center.
        # grasp_from=[0,0,1] → gripper approaches straight down, pre-position above the statue.
        # camera_up=[0,1,0] → fingers spread in world-y, gripping the widest axis of the body.
        target_pose = oscar_pose.add_bias([0, 0, 0.04], coord='world')
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0, 0, 1]),   # top-down approach (no table collision)
            np.array([1, 0, 0])    # camera_up rotated 90° around z vs [0,1,0]
        )
        grasp_idx = self.oscar.register_point(pose=grasp_pose, type='contact')
        self.move(self.atom.grasp_actor(
            self.oscar,
            contact_point_id=grasp_idx,
            is_close=False,
        ))
        self.move(self.atom.close_gripper(0.0, depth_threshold=7))
        
    def _play_once(self):
        self.move(self.atom.close_gripper(0.0, depth_threshold=7))
        self.origin_inhand_pose = self.oscar.get_pose().rebase(self.atom.get_arm_pose())
        # Move directly to above the pad — cuRobo plans a collision-free arc,
        # the lifting happens naturally as part of this motion
        pad_pose = self.stand.get_pose()
        ee_now = self.atom.get_arm_pose()
        above_pad = Pose([pad_pose.p[0], pad_pose.p[1], ee_now.p[2] + 0.1], ee_now.q)
        self.move(self.atom.move_to_pose(above_pad),time_dilation_factor=0.5)
        self.move(self.atom.close_gripper(0.0, depth_threshold=7))
        self.delay(30, is_save=True)
        self.move(self.atom.close_gripper(0.0, depth_threshold=30))
        self.move(self.atom.move_by_displacement(z=-0.05), time_dilation_factor=0.5)
        self.move(self.atom.move_by_displacement(z=-0.05), time_dilation_factor=0.5)
        # Relax grip and lower straight onto the pad
        self.move(self.atom.open_gripper())

    """def check_early_stop(self):
        ee_pos = self.atom.get_arm_pose().p
        oscar_pos = self.oscar.get_pose().p

        not_in_original = np.linalg.norm(oscar_pos - self._oscar_initial_pos) > 0.05
        far_from_arm = np.linalg.norm(oscar_pos - ee_pos) > 0.05

        if not_in_original and far_from_arm:
            return True
        return False"""

    def check_success(self):
        pad_pose = self.stand.get_pose()
        oscar_pos = self.oscar.get_pose().p
        in_zone = ( pad_pose.p[2] + 0.08 > oscar_pos[2] > pad_pose.p[2] + 0.03
                   and np.all(np.abs(oscar_pos[:2] - pad_pose.p[:2]) < 0.05))
        print("/" *80, oscar_pos[2] -  pad_pose.p[2])
        self._success_steps = (self._success_steps + 1) if in_zone else 0
        return in_zone
