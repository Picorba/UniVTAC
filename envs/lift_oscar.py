from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 15
    use_adaptive_grasp = True

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
    
    def create_actors(self):
        wall_pose = Pose([0.75, 0.0, 0.005], [1, 0, 0, 0])
        oscar_pose = wall_pose.add_bias([-0.18, 0.0, 0.03]).add_rotation([0, 0, np.pi/2])

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
            name='oscar',
            asset_path="oscar_detailled.usd",
            pose=oscar_pose,
            density=1e5
        )
    def _reset_actors(self):
        oscar_offset = self.create_noise([0.01, 0.05, 0.0], [0, 0, np.pi/18])
        oscar_pose = self.wall.get_pose().add_bias([-0.18, 0.0, 0.03]).add_rotation([0, 0, np.pi/2]).add_offset(oscar_offset)
        self.oscar.set_pose(oscar_pose)

    def pre_move(self):
        self.delay(10)

        oscar_pose = self.oscar.get_pose()
        target_pose = oscar_pose.add_bias([0.0, 0, -0.01], coord='world')
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
        self.target_pose = self.wall.get_pose().add_bias([0.0, 0, -0.01])
        
    def _play_once(self):
        self.move(self.atom.close_gripper())
        self.move(self.atom.move_by_displacement(z=0.1))
        self.delay(50, is_save=True)

    def check_mid_success(self):
        rel_pose = self.oscar.get_pose().rebase(self.target_pose)
        return rel_pose[0] > -0.01
    
    def check_early_stop(self):
        rel_pose = self.oscar.get_pose().rebase(self.target_pose)
        if self.take_action_cnt > 300 and np.abs(np.dot(rel_pose.to_transformation_matrix()[:3, 0], np.array([-1, 0, 0]))) > 0.99:
            return True
        return False

    def check_success(self):
        rel_pose = self.oscar.get_pose().rebase(self.target_pose)
        return rel_pose[0] > -0.02 and np.all(np.abs(rel_pose[1:3]) < np.array([0.1, 0.001])) \
            and np.abs(np.dot(rel_pose.to_transformation_matrix()[:3, 0], np.array([0, 0, 1]))) > 0.99