from ._base_task import *
import numpy as np

SORTED_ORDER = ['cylinder', 'prism_triangle', 'cube', 'star', 'prism_hexagon']
USD_MAP = {
    'cylinder':      'Cylinder.usd',
    'prism_triangle':'Prism_triangle.usd',
    'cube':          'Cube.usd',
    'star':          'Star.usd',
    'prism_hexagon': 'Prism_hexagon.usd',
}

# 5 evenly-spaced target y-positions (left-to-right = fewest→most sides)
Y_OFFSET = 0.25
TARGET_Y_OFFSETS = np.array([-Y_OFFSET*2, -Y_OFFSET, 0.0, Y_OFFSET, Y_OFFSET*2])

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 3000
    adaptive_grasp_depth_threshold = 10
    use_adaptive_grasp = True

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        wall_pose = Pose([0.75, 0.0, 0.005], [1, 0, 0, 0])
        self.wall = self._actor_manager.add_from_usd_file(
            name='wall', asset_path='Wall.usd', pose=wall_pose, density=1e5
        )
        base = wall_pose.add_bias([-0.18, 0.0, 0.08])
        self.shapes: dict[str, Actor] = {}
        for i, name in enumerate(SORTED_ORDER):
            pose = Pose(
                base.p + np.array([0.0, float(TARGET_Y_OFFSETS[i]), 0.0]),
                base.q
            )
            self.shapes[name] = self._actor_manager.add_from_usd_file(
                name=name, asset_path=USD_MAP[name], pose=pose, density=1e5
            )

    def _reset_actors(self):
        base = self.wall.get_pose().add_bias([-0.18, 0.0, 0.08])
        # Shuffle the y slots so shapes start in a random order
        shuffled_y = TARGET_Y_OFFSETS[self.rng.permutation(len(SORTED_ORDER))]
        for i, name in enumerate(SORTED_ORDER):
            noise = self.create_noise([0.005, 0.005, 0.0], [0, 0, np.pi / 12])
            pose = Pose(
                base.p + np.array([0.0, float(shuffled_y[i]), 0.0]),
                base.q
            ).add_offset(noise)
            self.shapes[name].set_pose(pose)

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(1.0))

    def _grasp_shape(self, actor: Actor):
        """Approach actor with a top-down grasp (gripper open)."""
        actor_pose = actor.get_pose()
        grasp_pose = construct_grasp_pose(
            actor_pose.p,
            np.array([0, 0, 1]),
            np.array([1, 0, 0])
        )
        grasp_idx = actor.register_point(pose=grasp_pose, type='contact')
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.grasp_actor(
            actor, contact_point_id=grasp_idx, is_close=False, pre_dis=0.15
        ))

    def _play_once(self):
        base = self.wall.get_pose().add_bias([-0.18, 0.0, 0.08])

        for i, name in enumerate(SORTED_ORDER):
            actor = self.shapes[name]
            target_y = float(TARGET_Y_OFFSETS[i])

            # Already in the correct slot — skip
            if np.abs(actor.get_pose().p[1] - target_y) < 0.02:
                continue

            # 1. Approach and close gripper
            self._grasp_shape(actor)
            self.move(self.atom.close_gripper())

            # 2. Lift clear of the other objects
            self.move(self.atom.move_by_displacement(z=0.12))

            # 3. Translate in y to align with the target slot
            dy = target_y - actor.get_pose().p[1]
            self.move(self.atom.move_by_displacement(y=dy))

            # 4. Lower back to resting height and release
            self.move(self.atom.move_by_displacement(z=-0.10))
            self.move(self.atom.open_gripper())
            self.delay(20)

    def check_success(self):
        tol = 0.05
        for i, name in enumerate(SORTED_ORDER):
            actor = self.shapes[name]
            target_y = float(TARGET_Y_OFFSETS[i])
            if np.abs(actor.get_pose().p[1] - target_y) > tol:
                return False
        return True
