from ._base_task import *

FRUITS_CONFIGS: dict[str, dict] = {
    'avocado': {
        'asset_path': 'avocado.usd',
        'density': 10,
    },
    'strawberry': {
        'asset_path': 'strawberry.usd',
        'density': 10,
    },
    'carrot': {
        'asset_path': 'carrot.usd',
        'density': 10,
    },
    'Apple': {
        'asset_path': 'apple.usd',
        'density': 10,
    },
    'banana': {
        'asset_path': 'banana.usd',
        'density': 10,
    },
}
BOWL_ASSET_PATH = "bowl.usd"
BOWL_BASE_POSE = Pose([0.5, 0, 0.041], [0.7071, 0, 0, 0.7071])
N_OBJECTS_PER_EPISODE = 3

@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    step_lim: int              = 1500
    use_force_grasp: bool      = True
    squeeze_force: float       = 5.0
    squeeze_steps: int         = 30
    n_objects: int             = N_OBJECTS_PER_EPISODE

class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self._fruits: dict[str, Actor] = {}
        for fruit in FRUITS_CONFIGS.keys():
            self.oscar = self._actor_manager.add_from_usd_file(
                name=fruit,
                asset_path=FRUITS_CONFIGS[fruit]["asset_path"],
                # pose=TODO,
                density=FRUITS_CONFIGS[fruit]["density"],
            )
        self.bowl = self._actor_manager.add_from_usd_file(
            name="bowl",
            asset_path=BOWL_ASSET_PATH,
            pose=BOWL_BASE_POSE,
        )

    def _reset_actors(self):
        nb_object = self.rng.choice(N_OBJECTS_PER_EPISODE, replace=False)
        select_fruits_in_scene()
        replace_bowl()
        replace_fruits_in_scene()
        replace_fruits_outscene()

    def pre_move(self):
        self.move(self.atom.open_gripper(pos=1.0))

    def _print_effort(self, label: str):
        effort = self._robot_manager.get_gripper_effort()
        print(
            f"[{label}]  left={effort[0].item():+.2f} N  right={effort[1].item():+.2f} N"
        )

    def _play_once(self):
        move_on_top(target_object)
        pick(object = target)
        move_on_top(target_place)
        drop()

    def check_success(self):
        is_target_in_bowl = False
        return is_target_in_bowl
