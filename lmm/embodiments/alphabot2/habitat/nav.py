"""Privileged navigation for AlphaBot 2 in Habitat (qualification / expert only, never a policy).

The follower plans a shortest path on the AlphaBot navmesh (radius 0.38 m) and
tracks it with differential-drive (v, omega) commands issued through the same
atomic ``AlphaBot2JointMobileAction`` the policy uses -- the base is never
teleported. Arms are held in the navigation tuck.

Collision accounting (``scene_contacts``): after each step a discrete
collision pass lists every contact point involving the robot. Contacts of
chassis/wheel links within 3 cm of the floor are the robot standing on the
floor ("support"); everything else is a robot-scene collision.
"""
from __future__ import annotations

import math

import magnum as mn
import numpy as np

from . import env as HE
from .alphabot2_robot import BASE_LINK_NAMES, NAV_TUCK


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose2d(env):
    """(x, z) position and yaw of the robot root in the Habitat world (yaw about +y, 0 = +x)."""
    T = HE.robot(env).sim_obj.transformation
    f = T.transform_vector(mn.Vector3(1, 0, 0))
    p = T.translation
    return np.array([p[0], p[2]]), math.atan2(-f[2], f[0]), np.array(p)


def yaw_to(src_xz, dst_xz):
    d = np.asarray(dst_xz) - np.asarray(src_xz)
    return math.atan2(-d[1], d[0])


def scene_contacts(env):
    rob = HE.robot(env)
    sim = env.sim
    sim.perform_discrete_collision_detection()
    rid = rob.sim_obj.object_id
    floor_y = float(rob.sim_obj.translation[1])
    names = {i: rob.sim_obj.get_link_name(i) for i in rob.sim_obj.get_link_ids()}
    coll, support = [], 0
    for c in sim.get_physics_contact_points():
        if (c.object_id_a == rid) == (c.object_id_b == rid):
            continue                                   # not the robot, or self-contact (checked in the smoke)
        a_is_robot = c.object_id_a == rid
        link = names.get(c.link_id_a if a_is_robot else c.link_id_b, "?")
        other = c.object_id_b if a_is_robot else c.object_id_a
        p = c.position_on_b_in_ws if a_is_robot else c.position_on_a_in_ws
        if link in BASE_LINK_NAMES and float(p[1]) < floor_y + 0.03:
            support += 1
            continue
        coll.append({"link": link, "other": int(other), "depth_m": float(c.contact_distance), "y": float(p[1])})
    return coll, support


def tuck_action(env, v=0.0, w=0.0):
    p = HE.robot(env).proprio16()
    return np.r_[NAV_TUCK["left"], p[7], NAV_TUCK["right"], p[15], v, w]


def move_arms_to_tuck(env, steps=15):
    rob = HE.robot(env)
    p0 = rob.proprio16()
    tgt = tuck_action(env)
    for i in range(steps):
        u = (i + 1) / steps
        a = p0.copy()
        a[0:7] = p0[0:7] + (tgt[0:7] - p0[0:7]) * u
        a[8:15] = p0[8:15] + (tgt[8:15] - p0[8:15]) * u
        HE.step18(env, np.r_[a, 0.0, 0.0])


def plan(env, goal_xyz):
    import habitat_sim
    pf = env.sim.pathfinder
    sp = habitat_sim.ShortestPath()
    sp.requested_start = HE.robot(env).sim_obj.translation
    sp.requested_end = mn.Vector3(*goal_xyz)
    ok = pf.find_path(sp)
    return ok, [np.array(p) for p in sp.points], float(sp.geodesic_distance)


def _densify(pts, ds=0.05):
    xz = [np.array([p[0], p[2]]) for p in pts]
    out = [xz[0]]
    for a, b in zip(xz[:-1], xz[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return np.array(out)


def follow_with(env, goal_xyz, step_vw, **kw):
    """``follow`` with a caller-supplied ``step_vw(env, v, w)`` (e.g. the expert's recording step)."""
    return follow(env, goal_xyz, step_vw=step_vw, **kw)


def follow(env, goal_xyz, face_xz=None, max_steps=1500, v_max=0.4, w_max=0.8, arrive_m=0.12, on_step=None,
           step_vw=None):
    """Drive to ``goal_xyz`` along the navmesh shortest path (then optionally face ``face_xz``).

    Segment follower for a differential base: Detour's shortest path hugs the navmesh edge at
    convex corners, so any corner-cutting chord leaves the mesh and the navmesh filter stops the
    base (measured). Instead: rotate in place to face the next path vertex, drive the straight
    segment (on-mesh by construction) with a small heading correction, and only keep rolling
    through bends shallower than 0.2 rad. Blocked motion (< 30 % of the commanded displacement
    for 5 steps) -> back off 0.1 m and replan."""
    ok, pts, geo = plan(env, goal_xyz)
    st = {"path_found": bool(ok and len(pts) > 0 and np.isfinite(geo)), "geodesic_m": geo, "steps": 0,
          "collision_steps": 0, "collision_links": {}, "max_depth_m": 0.0, "navmesh_filtered_m": 0.0,
          "travelled_m": 0.0, "replans": 0, "backoffs": 0, "rotate_steps": 0, "_step_vw": step_vw}
    if not st["path_found"]:
        st.pop("_step_vw")
        return st
    goal_xz = np.array([goal_xyz[0], goal_xyz[2]])
    V = [np.array([p[0], p[2]]) for p in pts][1:] or [goal_xz]
    blocked = 0
    while st["steps"] < max_steps and V:
        xz, yaw, _ = pose2d(env)
        if float(np.linalg.norm(goal_xz - xz)) < arrive_m:
            break
        tgt = V[0]
        d = float(np.linalg.norm(tgt - xz))
        last = len(V) == 1
        if d < (arrive_m if last else 0.06):
            V.pop(0)
            continue
        err = wrap(yaw_to(xz, tgt) - yaw)
        if abs(err) > 0.2:
            v, w = 0.0, float(np.clip(1.5 * err, -w_max, w_max))
            st["rotate_steps"] += 1
        else:
            # roll through a shallow bend at the next vertex instead of stopping at it
            nxt = V[1] if len(V) > 1 else None
            bend = abs(wrap(yaw_to(tgt, nxt) - yaw_to(xz, tgt))) if nxt is not None else math.pi
            slow = d if bend > 0.2 else d + 0.3
            v = float(min(v_max, 1.0 * slow + 0.04))
            w = float(np.clip(2.0 * err, -w_max, w_max))
        _step(env, v, w, st, on_step)
        xz2 = pose2d(env)[0]
        moved = float(np.linalg.norm(xz2 - xz))
        st["travelled_m"] += moved
        blocked = blocked + 1 if (v > 0.05 and moved < 0.3 * v * HE.ACTION_DT_S) else 0
        if blocked >= 5:
            for _ in range(5):
                _step(env, -0.2, 0.0, st, on_step)
            st["backoffs"] += 1
            ok, pts, _ = plan(env, goal_xyz)
            if not ok or not pts or st["replans"] >= 8:
                break
            V = [np.array([p[0], p[2]]) for p in pts][1:] or [goal_xz]
            st["replans"] += 1
            blocked = 0
    xz, yaw, _ = pose2d(env)
    st["arrival_dist_m"] = float(np.linalg.norm(goal_xz - xz))
    st["arrived"] = st["arrival_dist_m"] < arrive_m + 0.03
    if face_xz is not None:
        for _ in range(120):
            xz, yaw, _ = pose2d(env)
            err = wrap(yaw_to(xz, face_xz) - yaw)
            if abs(err) < 0.03:
                break
            _step(env, 0.0, float(np.clip(1.5 * err, -w_max, w_max)), st, on_step)
        xz, yaw, _ = pose2d(env)
        st["facing_err_rad"] = float(abs(wrap(yaw_to(xz, face_xz) - yaw)))
    st.pop("_step_vw", None)
    return st


def _step(env, v, w, st, on_step):
    obs = st["_step_vw"](env, v, w) if st.get("_step_vw") else HE.step18(env, tuck_action(env, v, w))
    st["steps"] += 1
    st["navmesh_filtered_m"] += HE.action_obj(env).last_info["base"]["navmesh_filtered_m"]
    coll, _ = scene_contacts(env)
    if coll:
        st["collision_steps"] += 1
        for c in coll:
            st["collision_links"][c["link"]] = st["collision_links"].get(c["link"], 0) + 1
            st["max_depth_m"] = min(st["max_depth_m"], c["depth_m"])
    if on_step is not None:
        on_step(obs)
    return obs
