from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.utils import configclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask


@configclass
class ObjectContactSensorCfg:
    """Per-actor contact sensor configuration."""
    name: str = 'contact'
    prim_path: str = None
    filter_prim_paths_expr: list = None  # None = report all contacts
    history_length: int = 1


class ContactSensorManager:
    """Manages Isaac Lab ContactSensors attached to rigid actors.

    Only works with RigidActor objects (PhysX rigid bodies created with
    activate_contact_sensors=True). UIPC Actor objects use a different
    physics backend and cannot be monitored this way.
    """

    def __init__(self, cfg_list: list[ObjectContactSensorCfg], task: 'BaseTask'):
        self.scene = task.scene
        self.sensors: dict[str, ContactSensor] = {}
        for cfg in cfg_list:
            self._add_sensor(cfg)

    def _add_sensor(self, cfg: ObjectContactSensorCfg):
        sensor_cfg = ContactSensorCfg(
            prim_path=cfg.prim_path,
            filter_prim_paths_expr=cfg.filter_prim_paths_expr or [],
            history_length=cfg.history_length,
            update_period=0.0,
        )
        sensor = ContactSensor(sensor_cfg)
        sensor._initialize_impl()
        sensor._is_initialized = True
        self.scene.sensors[f'contact_{cfg.name}'] = sensor
        self.sensors[cfg.name] = sensor

    def get_observations(self) -> dict:
        """Return {actor_name: {'net_force': Tensor(3), 'magnitude': Tensor(scalar)}}."""
        obs = {}
        for name, sensor in self.sensors.items():
            # net_forces_w: (num_envs, num_bodies, 3)
            net_force = sensor.data.net_forces_w[0, 0]
            obs[name] = {
                'net_force': net_force,
                'magnitude': net_force.norm(),
            }
        return obs
