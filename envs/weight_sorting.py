from ._base_task import *
import numpy as np

# Three visually identical cylinders (Bar_Cylinder.usd) with different densities.
# The robot lifts each, uses gripper effort as a mass proxy, ranks them,
# then places them left-to-right in ascending mass order.

_MASSES_G = [100, 300, 600]          # grams — easy to distinguish by force
_CYL_VOLUME = np.pi * 0.02**2 * 0.10  # Bar_Cylinder approx: r=2cm h=10cm
_DENSITIES = [m * 1e-3 / _CYL_VOLUME for m in _MASSES_G]

# Starting positions: three cylinders in a row along y
_CYL_X = 0.55
_CYL_Z = 0.02
_CYL_Y = [-0.15, 0.0, 0.15]

# Destination pads: shifted toward robot along x
_PAD_X = 0.40
_PAD_Y = [-0.15, 0.0, 0.15]

_LIFT_Z = 0.18
_HOLD_STEPS = 25  # sim steps to hold at lift height before reading effort


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim: int = 2000
    use_force_grasp: bool = True
    grasp_force: float = 12.0


class Task(BaseTask):

    def create_actors(self):
        self._cylinders: list[Actor] = []
        for i, density in enumerate(_DENSITIES):
            cyl = self._actor_manager.add_from_usd_file(
                name=f'cylinder_{i}',
                asset_path='Bar_Cylinder.usd',
                pose=Pose([_CYL_X, _CYL_Y[i], _CYL_Z], [1, 0, 0, 0]),
                density=density,
            )
            self._cylinders.append(cyl)

        self._pads: list[Actor] = []
        for i in range(3):
            pad = self._actor_manager.add_from_usd_file(
                name=f'pad_{i}',
                asset_path='GreenPad.usd',
                pose=Pose([_PAD_X, _PAD_Y[i], 0.005], [1, 0, 0, 0]),
                density=1e5,
            )
            self._pads.append(pad)

    def _reset_actors(self):
        # Shuffle which density goes to which starting slot
        perm = self.rng.permutation(3)
        self._density_perm = perm  # perm[slot] = density_index
        for slot, density_idx in enumerate(perm):
            self._cylinders[slot].set_pose(
                Pose([_CYL_X, _CYL_Y[slot], _CYL_Z], [1, 0, 0, 0])
            )
        self._recorded_efforts = [None, None, None]
        self._placed_order = []   # slot indices in the order placed (ascending effort)
        self._success_steps = 0

    # ------------------------------------------------------------------

    def pre_move(self):
        self.delay(10)

    def _play_once(self):
        n = 3

        # Phase 1: weigh each cylinder
        print('[weight_sorting] Phase 1: weighing')
        for slot in range(n):
            cyl = self._cylinders[slot]
            cyl.cfg.contact_points.clear()
            contact_idx = cyl.register_point(pose=cyl.get_pose(), type='contact')
            self.move(self.atom.grasp_actor(cyl, contact_point_id=contact_idx, is_close=False))
            self.move(self.atom.close_gripper())
            self.move(self.atom.move_by_displacement(z=_LIFT_Z))
            self.delay(_HOLD_STEPS, is_save=True)

            effort = self._robot_manager.get_gripper_effort()
            proxy = (abs(effort[0].item()) + abs(effort[1].item())) / 2.0
            self._recorded_efforts[slot] = proxy
            print(f'  slot {slot}: effort_proxy={proxy:.2f} N'
                  f'  (true_density={_DENSITIES[self._density_perm[slot]]:.0f} kg/m³)')

            self.move(self.atom.move_by_displacement(z=-_LIFT_Z))
            self.move(self.atom.open_gripper())
            self.delay(15)

        # Phase 2: place in ascending effort order
        print('[weight_sorting] Phase 2: placing')
        ranked_slots = sorted(range(n), key=lambda s: self._recorded_efforts[s])
        for dest_idx, slot in enumerate(ranked_slots):
            cyl = self._cylinders[slot]
            cyl.cfg.contact_points.clear()
            contact_idx = cyl.register_point(pose=cyl.get_pose(), type='contact')
            self.move(self.atom.grasp_actor(cyl, contact_point_id=contact_idx, is_close=False))
            self.move(self.atom.close_gripper())
            self.move(self.atom.place_actor(cyl, self._pads[dest_idx].get_pose(), is_open=True))
            self._placed_order.append(slot)
            print(f'  placed slot {slot} → pad {dest_idx}')

    def check_success(self) -> bool:
        if len(self._placed_order) < 3:
            return False
        # True ascending order by density index
        true_rank = sorted(range(3), key=lambda s: _DENSITIES[self._density_perm[s]])
        correct = sum(p == t for p, t in zip(self._placed_order, true_rank))
        rank_acc = correct / 3
        success = rank_acc >= 1.0
        print(f'[weight_sorting] rank_accuracy={rank_acc:.2f}  success={success}')
        self._success_steps = (self._success_steps + 1) if success else 0
        return success
