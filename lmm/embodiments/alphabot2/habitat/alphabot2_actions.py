"""AlphaBot 2 Habitat task action and proprio sensor (registered at import).

``AlphaBot2JointMobileAction`` applies ONE logical 10 Hz policy transition:
the complete ``JointMobileAction18`` -- left arm (7 vendor joints), right arm
(7), left and right gripper closed fractions, base forward velocity v (m/s) and
yaw rate omega (rad/s, counter-clockwise). All targets are staged inside this
single ``step()``; the environment then advances ONCE (``RearrangeSim.step``:
``ac_freq_ratio`` physics substeps + one render). There is no arm-step /
base-step split (the WXAI defect).

Base: (v, omega) are physical units. They are integrated over the action
interval ``action_dt`` (0.1 s) in ``base_substeps`` pieces, each filtered by
the navmesh with sliding (``pathfinder.try_step``), as habitat-lab's
BaseVelAction does for one piece -- but with the interval made explicit
instead of habitat's ``1/ctrl_freq`` (which ran at 120 Hz under a 0.032 s
env step for the old embodiments). No normalized [-1, 1] input.

Grasp (OVMM is kinematic_mode: Habitat's magic snap is the simulator's grasp
mechanism). Per hand, with its own RearrangeGraspManager (index 0 = left,
1 = right): a hand commanded to close (fraction >= ``close_threshold``) that
holds nothing snaps the task object only if the object's centre lies inside
the hand's closing region (pad midpoint within the object's bounding box
expanded by ``grasp_margin``) and the object's width across the fingers fits
the measured pad gap (1.5-7.8 cm). The fingers then stop at the object width. Opening
releases. A second hand cannot take an object already held. Every grasp /
release is logged as a simulator event (``last_info``) and never enters the
policy observation.
"""
from __future__ import annotations

import math

import habitat_sim
import magnum as mn
import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.actions.articulated_agent_action import ArticulatedAgentAction
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface

from .alphabot2_robot import GRIPPER_OPEN_RAD, SIDES

#: EG2-4C2 pad-SURFACE gap vs gripper joint, measured from the xacro collision meshes
#: (gripper_gap.py): 1.4 cm fully closed ... 8.0 cm fully open, linear in between.
_GAP_Q = np.array([0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.82])
_GAP_M = np.array([0.0142, 0.0231, 0.0324, 0.0422, 0.0521, 0.0622, 0.0722, 0.0804])
MIN_GRASP_WIDTH_M, MAX_GRASP_WIDTH_M = 0.015, 0.078


def q_for_aperture(width_m: float) -> float:
    """Gripper joint at which the pads close on an object of ``width_m`` across the fingers."""
    return float(np.interp(width_m, _GAP_M, _GAP_Q))


@registry.register_task_action
class AlphaBot2JointMobileAction(ArticulatedAgentAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self.action_dt = float(config.get("action_dt", 0.1))
        self.base_substeps = int(config.get("base_substeps", 5))
        self.v_max = float(config.get("v_max", 1.0))
        self.w_max = float(config.get("w_max", 2.8))
        self.close_threshold = float(config.get("close_threshold", 0.5))
        self.grasp_margin = float(config.get("grasp_margin", 0.005))
        self.vel = habitat_sim.physics.VelocityControl()
        self.vel.controlling_lin_vel = self.vel.controlling_ang_vel = True
        self.vel.lin_vel_is_local = self.vel.ang_vel_is_local = True
        self.last_info: dict = {}
        self.events: list = []

    @property
    def action_space(self):
        return spaces.Dict({self._action_arg_prefix + "alphabot2_action":
                            spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)})

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.events = []
        self.last_info = {}

    # ---------------------------------------------------------------- base
    def _base(self, v: float, w: float) -> dict:
        robot = self.cur_articulated_agent
        v_c, w_c = float(np.clip(v, -self.v_max, self.v_max)), float(np.clip(w, -self.w_max, self.w_max))
        self.vel.linear_velocity = mn.Vector3(v_c, 0.0, 0.0)          # root +x = forward
        self.vel.angular_velocity = mn.Vector3(0.0, w_c, 0.0)         # root +y = up -> +w counter-clockwise
        start = robot.sim_obj.transformation
        filtered = 0.0
        if v_c != 0.0 or w_c != 0.0:
            dt = self.action_dt / self.base_substeps
            for _ in range(self.base_substeps):
                tr = robot.sim_obj.transformation
                rs = habitat_sim.RigidState(mn.Quaternion.from_matrix(tr.rotation()), tr.translation)
                tgt = self.vel.integrate_transform(dt, rs)
                # sliding navmesh filter (pathfinder.try_step), NOT sim.step_filter: OVMM sets
                # allow_sliding=False, where try_step_no_sliding stops the base dead the moment it
                # grazes the boundary -- and Detour paths run exactly along it (measured: blocked
                # after ~1 m on corridor segments). The boundary is already the full chassis radius
                # (0.38 m) away from real geometry, so sliding along it involves no contact; real
                # robot-scene contacts are measured separately every step.
                end = self._sim.pathfinder.try_step(rs.translation, tgt.translation)
                filtered += float((end - tgt.translation).length())
                robot.sim_obj.transformation = mn.Matrix4.from_(tgt.rotation.to_matrix(), end)
        end = robot.sim_obj.transformation
        return {"v_cmd": v, "w_cmd": w, "v_applied": v_c, "w_applied": w_c,
                "disp_m": float((end.translation - start.translation).length()),
                "navmesh_filtered_m": filtered}

    # ---------------------------------------------------------------- grasp
    def _task_objects(self):
        rom = self._sim.get_rigid_object_manager()
        return [(i, rom.get_object_by_id(oid)) for i, oid in enumerate(self._sim.scene_obj_ids)]

    def _grasp_candidate(self, side_idx: int):
        robot = self.cur_articulated_agent
        ee = robot.ee_transform(side_idx)
        p = np.array(ee.translation)
        closing = np.array(ee.transform_vector(mn.Vector3(1.0, 0.0, 0.0)))    # recomputed below from the pads
        pads = [robot.link_frame(f"{SIDES[side_idx]}_{n}")[:3, 3] for n in ("left_pad", "right_pad")]
        closing = (pads[0] - pads[1]) / max(1e-9, np.linalg.norm(pads[0] - pads[1]))
        best = None
        for idx, obj in self._task_objects():
            bb = obj.root_scene_node.cumulative_bb
            T = obj.transformation
            local = np.array(T.inverted().transform_point(mn.Vector3(*p)))
            lo, hi = np.array(bb.min), np.array(bb.max)
            inside = bool(np.all(local >= lo - self.grasp_margin) and np.all(local <= hi + self.grasp_margin))
            if not inside:
                continue
            # object width across the fingers: extent of its box along the closing axis
            R = np.array(T.rotation())
            half = (hi - lo) / 2
            width = float(2 * np.sum(np.abs((R.T @ closing)) * half))
            d = float(np.linalg.norm(np.array(T.transform_point(mn.Vector3(*((lo + hi) / 2)))) - p))
            if best is None or d < best[2]:
                best = (idx, obj, d, width)
        return best

    def _gripper(self, side_idx: int, frac: float) -> dict:
        robot = self.cur_articulated_agent
        side = SIDES[side_idx]
        mgrs = self._articulated_agent_mgr.grasp_mgrs
        mgr = mgrs[side_idx]
        info = {"cmd": frac}
        if frac >= self.close_threshold:
            if mgr.is_grasped:
                info["holding"] = True
                return info
            cand = self._grasp_candidate(side_idx)
            other = mgrs[1 - side_idx]
            if (cand is not None and MIN_GRASP_WIDTH_M <= cand[3] <= MAX_GRASP_WIDTH_M
                    and not (other.is_grasped and other.snap_idx == cand[1].object_id)):
                idx, obj, d, width = cand
                self._articulated_agent_mgr.active_grasp_manager_idx = side_idx
                mgr.snap_to_obj(obj.object_id, force=False)
                # keep the object exactly where it is relative to the hand: in kinematic mode
                # snap_to_obj never sets keep_T, and update_object_to_grasp would then teleport the
                # object's origin onto the pad midpoint with the hand's orientation every step
                mgr._keep_T = robot.ee_transform(side_idx).inverted() @ obj.transformation
                robot.set_gripper_q(side, q_for_aperture(width))
                info.update({"grasp_event": True, "object_index": idx, "object_width_m": width, "dist_m": d})
                self.events.append({"type": "grasp", "hand": side, "object_index": idx, "width_m": width})
            else:
                robot.set_gripper_q(side, robot.q_from_fraction(frac))
                info["grasp_event"] = False
        else:
            if mgr.is_grasped:
                self._articulated_agent_mgr.active_grasp_manager_idx = side_idx
                mgr.desnap()
                info["release_event"] = True
                self.events.append({"type": "release", "hand": side})
            robot.set_gripper_q(side, robot.q_from_fraction(frac))
        return info

    # ---------------------------------------------------------------- step
    def step(self, *args, **kwargs):
        a = np.asarray(kwargs[self._action_arg_prefix + "alphabot2_action"], np.float64).reshape(-1)
        if a.shape != (18,) or not np.isfinite(a).all():
            raise ValueError(f"AlphaBot2JointMobileAction needs a finite 18-D action, got {a.shape}")
        robot = self.cur_articulated_agent
        info = {"arm": {}, "grip": {}}
        # 1) both arms (kinematic joint targets)
        info["arm"]["left"] = robot.set_arm_q("left", a[0:7])
        info["arm"]["right"] = robot.set_arm_q("right", a[8:15])
        # 2) base, integrated over the whole action interval
        info["base"] = self._base(a[16], a[17])
        # 3) both grippers (grasp/release against the updated arm + base pose)
        info["grip"]["left"] = self._gripper(0, a[7])
        info["grip"]["right"] = self._gripper(1, a[15])
        # 4) held objects follow their hand
        for m in self._articulated_agent_mgr.grasp_mgrs:
            if m.is_grasped:
                m.update_object_to_grasp()
        self.last_info = info


@registry.register_sensor
class AlphaBot2ProprioSensor(UsesArticulatedAgentInterface, Sensor):
    """Canonical 16-D proprio [left 7, left closed fraction, right 7, right closed fraction]."""

    cls_uuid = "alphabot2_proprio"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        return self._sim.get_agent_data(self.agent_id).articulated_agent.proprio16().astype(np.float32)


def register_robot():
    """Make ``articulated_agent_type: AlphaBot2Robot`` resolvable without patching habitat-lab."""
    import habitat.tasks.rearrange.articulated_agent_manager as am
    from .alphabot2_robot import AlphaBot2Robot
    am.AlphaBot2Robot = AlphaBot2Robot
    return AlphaBot2Robot
