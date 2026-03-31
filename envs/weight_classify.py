from ._base_task import *
import numpy as np


CUBE_START_POSE = Pose([0.35, 0.0, 0.01], [1, 0, 0, 0])

DENSITY_LIGHT = 1000
DENSITY_HEAVY = 5000


@configclass
class TaskCfg(BaseTaskCfg):
    use_adaptive_grasp = False


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        """cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5"""
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        green_pose = Pose([0.4, 0.08, 0.01], [1, 0, 0, 0])
        orange_pose = Pose([0.4, -0.08, 0.01], [1, 0, 0, 0])

        self.green_pad = self._actor_manager.add_from_usd_file(
            name='green_pad',
            asset_path="GreenPad.usd",
            pose=green_pose,
        )
        self.orange_pad = self._actor_manager.add_from_usd_file(
            name='orange_pad',
            asset_path="OrangePad.usd",
            pose=orange_pose,
        )

        # light cube (density 1000) → green pad
        # heavy cube (density 5000) → orange pad
        self.light_cube = self._actor_manager.add_from_usd_file(
            name='light_cube',
            asset_path="PlainPrism.usd",
            pose=Pose([0.35, 1.0, 0.01], [1, 0, 0, 0]),
            density=DENSITY_LIGHT,
        )
        self.heavy_cube = self._actor_manager.add_from_usd_file(
            name='heavy_cube',
            asset_path="PlainPrism.usd",
            pose=Pose([0.35, -1.0, 0.01], [1, 0, 0, 0]),
            density=DENSITY_HEAVY,
        )

    def _reset_actors(self):
        self.choice = self.rng.choice(['light', 'heavy'])
        if self.choice == 'light':
            self.cube = self.light_cube
            self.target = self.green_pad
            self.other_target = self.orange_pad
        else:
            self.cube = self.heavy_cube
            self.target = self.orange_pad
            self.other_target = self.green_pad
        self.cube.set_pose(CUBE_START_POSE)
        self.domain_rand_params = {
            'cube_type': self.choice,
        }

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))

        target_pose = self.cube.get_pose().add_bias([0.0, 0.0, 0.04 + 0.01 * self.rng.random()])
        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        cid = self.cube.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.cube, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
        ))
        gripper_qpos = self.rng.uniform(0.0065, 0.0075) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos))
        self.move(self.atom.move_by_displacement(z=0.05))

        self.target_pose = self.target.get_pose().add_bias([0.0, 0.0, 0.015])

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.cube,
            target_pose=self.target_pose,
            pre_dis=0.0, dis=0.0,
            is_open=False
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        cube_pose = self.cube.get_pose().rebase(self.target_pose)
        return np.all(np.abs(cube_pose.p) < np.array([0.02, 0.02, 0.01])) and \
            np.dot(cube_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965  # 15°
