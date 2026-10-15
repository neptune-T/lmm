"""Per-frame navigation observability features of recorded expert episodes (DIAGNOSTICS ONLY, privileged).

For every navigation frame (stride STRIDE) of the given accepted episodes, from the recorded base pose
(metadata.npz base_xz_yaw) in the episode's own seeded world (HE.reset_to, the collection contract):
  visible            target object seen by the head camera: some of {bbox centre, 8 shrunk corners} projects
                     inside the 69 deg x 320x240 image in front of the camera and a ray from the camera to it
                     is not blocked by other geometry (the robot's own links are ignored) -- navigate_to_object
                     frames; on navigate_to_receptacle frames the target is the nearest goal receptacle point
  vis_points         how many of those 9 points are visible
  dist_target_m      euclidean base-to-target distance
  bearing_target     target direction relative to the base heading (rad, + = counter-clockwise, the sign of
                     a positive omega)
  bearing_waypoint   direction of the point 1 m ahead on the navmesh shortest path from the frame's pose to
                     where the expert ended that navigation stage (the expert follows such a path)
  geodesic_left_m    geodesic distance left to that stage end
  v, omega           the expert's labels; t, stage, steps_left (to the stage end)
The head camera is rigid on the base: its pose per frame = recorded base pose x the base->camera transform
measured once at reset. Nothing here is written into any dataset.

    UNITS_FILE=... OUT=... WORKER=k NWORKERS=n GPU=0 run_habitat.sh nav_obs_features.py
"""
import json, math, os, sys

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import fsutil as FS  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402

FS.protect_stdio()
NAV_STAGES = ("navigate_to_object", "navigate_to_receptacle")
STRIDE = int(os.environ.get("STRIDE", "2"))
HFOV = math.radians(69.0)
TX, TY = math.tan(HFOV / 2), math.tan(HFOV / 2) * 240 / 320
OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
items = [json.loads(l) for l in open(os.environ["UNITS_FILE"]) if l.strip()][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split="train")


def np4(M):
    return np.array([[M[c][r] for c in range(4)] for r in range(4)])


def root_at(x, z, yaw, y0, yaw0, R0):
    T = np.eye(4)
    a = yaw - yaw0
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T[:3, :3] = Ry @ R0
    T[:3, 3] = [x, y0, z]
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def bearing(xz, yaw, p):
    return wrap(math.atan2(-(p[2] - xz[1]), p[0] - xz[0]) - yaw)


def visible(cam_T, pts, target_ids, ignore):
    sim = env.sim
    Ti = np.linalg.inv(cam_T)
    o = cam_T[:3, 3]
    n = 0
    for p in pts:
        q = Ti @ np.r_[p, 1.0]
        if q[2] <= 0.05 or abs(q[0] / q[2]) > TX or abs(q[1] / q[2]) > TY:
            continue
        d = p - o
        L = float(np.linalg.norm(d))
        if L > 12.0:
            continue
        res = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(*o), mn.Vector3(*(d / L))), max_distance=L + 0.05)
        ok = True
        for h in res.hits:
            if h.object_id in ignore:
                continue
            ok = h.object_id in target_ids or h.ray_distance >= L - 0.08
            break
        n += int(ok)
    return n


fn = f"{OUT}/navobs_w{WK:02d}.jsonl"
done = set()
if os.path.exists(fn):
    done = {json.loads(l)["unit"] for l in open(fn) if l.strip()}
for it in items:
    unit, d = it["unit"], it["dir"]
    if unit in done:
        continue
    rec = {"unit": unit, "split_role": it.get("role"), "frames": []}
    try:
        e, r = HE.parse_unit(unit)
        HE.reset_to(env, e, r)
        HE.fix_start(env)
        sim, rob = env.sim, HE.robot(env)
        md = np.load(f"{d}/metadata.npz")
        act = np.load(f"{d}/lowdim.npz")["action"]
        base, stg = np.asarray(md["base_xz_yaw"], float), [str(s) for s in md["stage"]]
        T0 = np4(rob.sim_obj.transformation)
        xz0, yaw0, _ = NAV.pose2d(env)
        if not np.allclose([xz0[0], xz0[1], yaw0], base[0], atol=1e-3):
            raise RuntimeError(f"start pose mismatch {xz0, yaw0} vs {base[0]}")
        X = np.linalg.inv(T0) @ rob.link_frame("head_camera")            # base -> head optical frame
        idx, goals = sim.get_targets()
        obj = sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(idx[0])])
        bb = obj.root_scene_node.cumulative_bb
        Tobj = np4(obj.transformation)
        c = (np.array(bb.min) + np.array(bb.max)) / 2
        corners = [c + 0.8 * (np.array([bx, by, bz]) - c) for bx in (bb.min[0], bb.max[0]) for by in (bb.min[1], bb.max[1])
                   for bz in (bb.min[2], bb.max[2])]
        obj_pts = [(Tobj @ np.r_[p, 1.0])[:3] for p in [c] + corners]
        goal_pts = [np.asarray(g, float) for g in np.asarray(goals).reshape(-1, 3)]
        ignore = {rob.sim_obj.object_id, *rob.sim_obj.link_object_ids.keys()}
        tid = {obj.object_id}
        runs, s0 = [], 0                                                  # navigation stage runs [start, end)
        for t in range(1, len(stg) + 1):
            if t == len(stg) or stg[t] != stg[s0]:
                if stg[s0] in NAV_STAGES:
                    runs.append((s0, t, stg[s0]))
                s0 = t
        for a0, a1, name in runs:
            end = base[a1 - 1]
            end_nav = sim.pathfinder.snap_point(mn.Vector3(end[0], T0[1, 3], end[1]))
            for t in range(a0, a1, STRIDE):
                x, z, yaw = base[t]
                Tc = root_at(x, z, yaw, T0[1, 3], yaw0, T0[:3, :3]) @ X
                if name == "navigate_to_object":
                    pts, tgt = obj_pts, obj_pts[0]
                    nvis = visible(Tc, pts, tid, ignore)
                else:
                    tgt = min(goal_pts, key=lambda g: math.hypot(g[0] - x, g[2] - z))
                    nvis = None
                sp = habitat_sim.ShortestPath()
                sp.requested_start = sim.pathfinder.snap_point(mn.Vector3(x, T0[1, 3], z))
                sp.requested_end = end_nav
                found = sim.pathfinder.find_path(sp)
                wp, geo = None, None
                if found and len(sp.points) >= 2:
                    geo = float(sp.geodesic_distance)
                    acc, prev = 0.0, np.array(sp.points[0])
                    wp = np.array(sp.points[-1])
                    for q in sp.points[1:]:
                        q = np.array(q)
                        seg = float(np.linalg.norm(q - prev))
                        if acc + seg >= 1.0:
                            wp = prev + (q - prev) * (1.0 - acc) / max(seg, 1e-9)
                            break
                        acc, prev = acc + seg, q
                rec["frames"].append({"t": t, "stage": name, "steps_left": a1 - t, "visible": None if nvis is None else nvis > 0,
                                      "vis_points": nvis, "dist_target_m": round(math.hypot(tgt[0] - x, tgt[2] - z), 3),
                                      "bearing_target": round(bearing((x, z), yaw, tgt), 4),
                                      "bearing_waypoint": None if wp is None else round(bearing((x, z), yaw, wp), 4),
                                      "geodesic_left_m": None if geo is None else round(geo, 3),
                                      "v": round(float(act[t, 16]), 4), "omega": round(float(act[t, 17]), 4)})
    except Exception as ex:                                               # recorded, never fatal
        rec["error"] = repr(ex)[:300]
    FS.append_line(fn, json.dumps(rec))
    print("NAVOBS", unit, len(rec["frames"]), rec.get("error", ""), flush=True)
env.close()
