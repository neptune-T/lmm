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
            # held object as its index in the episode's object list (raw simulator ids are re-assigned by every
            # hard world rebuild, so they differ between a recording and its fresh-reset replay)
            "held": [None if m._snapped_obj_id is None else
                     (list(sim.scene_obj_ids).index(m._snapped_obj_id) if m._snapped_obj_id in sim.scene_obj_ids else "other")
                     for m in sim.get_agent_data(0).grasp_mgrs]}


def _rot_deg(Ra, Rb) -> float:
    """Exact rotation angle between two rotation matrices, stable for small angles:
    ||Ra - Rb||_F = 2*sqrt(2)*sin(theta/2). (arccos of the trace on float32 matrices reads ~0.05 deg of noise.)"""
    d = np.linalg.norm(np.asarray(Ra, np.float64) - np.asarray(Rb, np.float64))
    return float(np.degrees(2.0 * np.arcsin(min(1.0, d / (2.0 * np.sqrt(2.0))))))


def divergence(a: dict, b: dict) -> dict:
    """Position and rotation errors reported SEPARATELY (never a mixed max over matrix entries):
    robot root position (m) / rotation (deg), joint proprio (max abs), every scene object's position (m)
    and rotation (deg) -- max over objects, with the worst object's index -- and held-object equality."""
    ra, rb = np.asarray(a["root"]), np.asarray(b["root"])
    oa, ob = np.asarray(a["objects"]).reshape(-1, 3, 4), np.asarray(b["objects"]).reshape(-1, 3, 4)
    pos = np.linalg.norm(oa[:, :, 3] - ob[:, :, 3], axis=1) if len(oa) else np.zeros(0)
    rot = np.array([_rot_deg(x[:, :3], y[:, :3]) for x, y in zip(oa, ob)]) if len(oa) else np.zeros(0)
    return {"root_pos_m": float(np.linalg.norm(ra[:3, 3] - rb[:3, 3])), "root_rot_deg": _rot_deg(ra[:3, :3], rb[:3, :3]),
            "proprio_max_abs": float(np.max(np.abs(a["proprio"] - b["proprio"]))),
            "objects_pos_max_m": float(pos.max()) if len(pos) else 0.0,
            "objects_rot_max_deg": float(rot.max()) if len(rot) else 0.0,
            "objects_pos_worst_idx": int(pos.argmax()) if len(pos) else None,
            "objects_rot_worst_idx": int(rot.argmax()) if len(rot) else None,
            "held_equal": a["held"] == b["held"]}


METRICS = ("root_pos_m", "root_rot_deg", "proprio_max_abs", "objects_pos_max_m", "objects_rot_max_deg")


def max_divergence(ds: list) -> dict:
    """Per-metric maximum over a trajectory of divergence() dicts (+ held equality at every step)."""
    if not ds:
        return None
    out = {k: max(d[k] for d in ds) for k in METRICS}
    out["held_equal"] = all(d["held_equal"] for d in ds)
    return out