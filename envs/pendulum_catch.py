from ._base_task import *
import os
import numpy as np



PENDULUM_BASE_POSE = Pose([0.55, 0.0, 0.03], [1, 0, 0, 0])
@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 7.5
    use_adaptive_grasp = False
    use_force_grasp: bool = True
    grasp_force: float = float(os.environ.get('LIFT_OSCAR_GRASP_FORCE', 10.0))  # Newtons
    DENSITY: int = int(os.environ.get('LIFT_OSCAR_DENSITY', 500))


class Task(BaseTask):
    def create_actors(self):
        self.pendulum = self._actor_manager.add_from_usd_file(
            name='cylinder', asset_path="Bar_Cylinder.usd", pose=PENDULUM_BASE_POSE, density=self.cfg.DENSITY
        )
        
    def _reset_actors(self):

        self.domain_rand_params = {
            'density': self.cfg.DENSITY,
        }

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.close_gripper(0.0))

    def _play_once(self):
        self.move(self.atom.close_gripper())
        self.delay(30, is_save=True)
        self.move(self.atom.open_gripper())
        self.delay(30, is_save=True)
        self.move(self.atom.close_gripper())
        

    def check_success(self):
        oscar_pos = self.pendulum.get_pose().p
        distance_hand_cylinder = 10
        straight = False
        return  distance_hand_cylinder < 0.5 and straight
