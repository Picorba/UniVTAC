from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 10
    use_adaptive_grasp = True

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
    
    def create_actors(self):
        wall_pose = Pose([0.75, 0.0, 0.005], [1, 0, 0, 0])
        oscar_pose = wall_pose.add_bias([-0.18, 0.0, 0.03]).add_rotation([0, np.pi/2, np.pi/2])

        self.wall = self._actor_manager.add_from_usd_file(
            name='wall',
            asset_path="Wall.usd",
            pose=wall_pose,
            density=1e5
        )
        self.oscar = self._actor_manager.add_from_usd_file(
            name='oscar',
            asset_path="oscar.usd",
            pose=oscar_pose,
            density=2e5
        )
    def _reset_actors(self):
        oscar_offset = self.create_noise([0.01, 0.05, 0.0], [0, 0, np.pi/18])
        oscar_pose = self.wall.get_pose().add_bias([-0.18, 0.0, 0.03]).add_rotation([0, np.pi/2, np.pi/2]).add_offset(oscar_offset)
        self.oscar.set_pose(oscar_pose)

    def pre_move(self):
        self.delay(10)

        oscar_pose = self.oscar.get_pose()
        target_pose = oscar_pose.add_bias([-0.05, 0, -0.011], coord='world')  # Shift in x so it grabs the head

        self.grasp_noise = self.create_noise(euler=[0, [-np.pi/12, 0.0], 0])
        target_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0, 0, 1]),
            np.array([1, 0, 0])
        ).add_offset(self.grasp_noise)
        grasp_idx = self.oscar.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(
            self.oscar,
            contact_point_id=grasp_idx,
            is_close=False,
            pre_dis=0.5
        ))
        self.target_pose = self.wall.get_pose().add_bias([-0.18, 0, 0.05])
        
    def _play_once(self):
        self.move(self.atom.close_gripper())
        self.move(self.atom.move_by_displacement(z=0.1))
        if not self.check_mid_success():
            # Oscar didn't follow the gripper — try an additional boost
            self.move(self.atom.move_by_displacement(z=0.05))
        self.move(self.atom.open_gripper(0.5))
        self.delay(30, is_save=False)

    def check_mid_success(self):
        # Oscar center starts at z≈0.035; confirm it is being lifted
        return self.oscar.get_pose()[2] > 0.08

    def check_early_stop(self):
        # Stop early if oscar has dropped back to near its start height after 300 steps
        if self.take_action_cnt > 300 and self.oscar.get_pose()[2] < 0.04:
            return True
        return False

    def check_success(self):
        # Oscar center (origin at middle) must be above 8cm — confirms it was lifted off
        print("/" * 80, self.oscar.get_pose()[2])
        return self.oscar.get_pose()[2] > 0.06