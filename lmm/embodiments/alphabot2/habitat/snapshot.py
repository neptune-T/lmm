"""Snapshot / restore of an AlphaBot 2 OVMM episode in Habitat (for LMM branching).

Captured (and restored) state:
* robot: root transform (base pose), all 26 joint positions and velocities
  (14 arm joints, both grippers incl. their mimic joints);
* task objects (``sim.scene_obj_ids``): transform, linear/angular velocity,
  motion type and collidability (a released object is DYNAMIC);
* grasp ownership per hand: each RearrangeGraspManager's snapped object id,
  ``keep_T`` (object pose in the hand), leave info; the active manager index;
* task state: step counters, picked-object index, and every OVMM measure's
  plain-data attributes (phase-success flags are cumulative, so a branch must
  inherit them), plus the AlphaBot action's grasp/release event log;
* policy-conditioning state is the caller's (e.g. CartesianExecutor.q /
  prev_quat) and is captured by ``capture(..., extra=...)``.

Bullet's internal solver caches are not restorable; in OVMM kinematic mode only
released (DYNAMIC) objects are integrated by physics, so that is where any
divergence would appear -- the qualification measures it.
"""
from __future__ import annotations

import copy

import magnum as mn
import numpy as np

from . import env as HE

_PLAIN = (bool, int, float, str, type(None), np.ndarray, np.floating, np.integer)


def _plain(v):
    if isinstance(v, _PLAIN):
        return True
    if isinstance(v, (list, tuple)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _plain(x) for k, x in v.items())
    return False


def _m4(M):
    return np.array([[M[c][r] for c in range(4)] for r in range(4)])


def _from_np(T):
    return mn.Matrix4([[float(T[r, c]) for r in range(4)] for c in range(4)])


def capture(env, extra=None) -> dict:
    sim, rob = env.sim, HE.robot(env)
    rom = sim.get_rigid_object_manager()
    ad = sim.get_agent_data(0)
    snap = {
        "root": _m4(rob.sim_obj.transformation),
        "q": np.asarray(rob.sim_obj.joint_positions, np.float64).copy(),
        "qd": np.asarray(rob.sim_obj.joint_velocities, np.float64).copy(),
        "objects": {},
        "grasp": [],
        "active_grasp": getattr(ad, "active_grasp_manager_idx", 0),
        "elapsed_steps": env._elapsed_steps,
        "task": {k: copy.deepcopy(v) for k, v in vars(env.task).items() if k.startswith("_") and _plain(v)},
        "measures": {},
        "action_events": copy.deepcopy(HE.action_obj(env).events),
        "extra": copy.deepcopy(extra),
    }
    for oid in sim.scene_obj_ids:
        o = rom.get_object_by_id(oid)
        snap["objects"][oid] = {"T": _m4(o.transformation), "lv": np.array(o.linear_velocity),
                                "av": np.array(o.angular_velocity), "motion": o.motion_type,
                                "collidable": o.collidable}
    for m in ad.grasp_mgrs:
        snap["grasp"].append({"obj": m._snapped_obj_id, "keep_T": None if m._keep_T is None else _m4(m._keep_T),
                              "marker": m._snapped_marker_id, "leave": copy.copy(m._leave_info)})
    for name, meas in env.task.measurements.measures.items():
        snap["measures"][name] = {k: copy.deepcopy(v) for k, v in vars(meas).items() if _plain(v)}
    return snap


def restore(env, snap):
    sim, rob = env.sim, HE.robot(env)
    rom = sim.get_rigid_object_manager()
    ad = sim.get_agent_data(0)
    rob.sim_obj.transformation = _from_np(snap["root"])
    rob.sim_obj.joint_positions = snap["q"].tolist()
    rob.sim_obj.joint_velocities = snap["qd"].tolist()
    for oid, s in snap["objects"].items():
        o = rom.get_object_by_id(oid)
        o.motion_type = s["motion"]
        o.collidable = s["collidable"]
        o.transformation = _from_np(s["T"])
        o.linear_velocity = mn.Vector3(*s["lv"])
        o.angular_velocity = mn.Vector3(*s["av"])
    for m, g in zip(ad.grasp_mgrs, snap["grasp"]):
        m._snapped_obj_id = g["obj"]
        m._keep_T = None if g["keep_T"] is None else _from_np(g["keep_T"])
        m._snapped_marker_id = g["marker"]
        m._leave_info = copy.copy(g["leave"])
    if hasattr(ad, "active_grasp_manager_idx"):
        ad.active_grasp_manager_idx = snap["active_grasp"]
    env._elapsed_steps = snap["elapsed_steps"]
    for k, v in snap["task"].items():
        setattr(env.task, k, copy.deepcopy(v))
    for name, attrs in snap["measures"].items():
        meas = env.task.measurements.measures.get(name)
        if meas is None:
            continue
        for k, v in attrs.items():
            setattr(meas, k, copy.deepcopy(v))
    HE.action_obj(env).events = copy.deepcopy(snap["action_events"])
    rob.update()                       # cameras follow the restored links
    return copy.deepcopy(snap["extra"])


def state_vector(env) -> dict:
    """Compact comparable state (for divergence measurements)."""
    sim, rob = env.sim, HE.robot(env)
    rom = sim.get_rigid_object_manager()
    return {"root": _m4(rob.sim_obj.transformation), "proprio": rob.proprio16(),
            "objects": np.concatenate([_m4(rom.get_object_by_id(o).transformation)[:3, :].ravel()
                                       for o in sim.scene_obj_ids]),
            "held": [m._snapped_obj_id for m in sim.get_agent_data(0).grasp_mgrs]}


def divergence(a: dict, b: dict) -> dict:
    return {"root_max_abs": float(np.max(np.abs(a["root"] - b["root"]))),
            "proprio_max_abs": float(np.max(np.abs(a["proprio"] - b["proprio"]))),
            "objects_max_abs": float(np.max(np.abs(a["objects"] - b["objects"]))),
            "held_equal": a["held"] == b["held"]}
