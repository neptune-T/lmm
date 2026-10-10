"""OVMM navigation-only qualification for AlphaBot 2 (manipulation disabled, arms tucked).

Real OVMM episodes, normal initial state. Per episode:
  start validity on the AlphaBot navmesh (radius 0.38 m) and start collisions;
  leg 1: start -> object interaction region (nearest navigable OVMM view point of
         a candidate object, geodesic on the AlphaBot navmesh), then face the object;
  leg 2: object region -> goal-receptacle region, then face the receptacle.
The base is driven only by (v, omega) through the atomic policy action.
Measured: path validity, arrival distance, facing error, collision steps and
links, navmesh filtering, travelled vs geodesic length, OVMM's own nav measures,
and how many OVMM view points (generated for Stretch, radius 0.3) are still
navigable for AlphaBot.

    EPISODES_PER_SCENE=2 MAX_SCENES=15 OUT=... run_habitat.sh nav_qual.py
"""
import json, os, sys, time
from collections import defaultdict

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/runtime/runs/nav_qual")
os.makedirs(OUT, exist_ok=True)
PER_SCENE = int(os.environ.get("EPISODES_PER_SCENE", "2"))
MAX_SCENES = int(os.environ.get("MAX_SCENES", "15"))
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
RADIUS = os.environ.get("NAV_RADIUS")
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "val"),
                  overrides=[f"habitat.simulator.agents.main_agent.radius={RADIUS}"] if RADIUS else [])
by_scene = defaultdict(list)
for e in env.episodes:
    by_scene[e.scene_id].append(e.episode_id)
scenes = sorted(by_scene)[:MAX_SCENES]
eps = [eid for s in scenes for eid in by_scene[s][:PER_SCENE]][WK::NW]
OVMM_KEYS = ("ovmm_nav_to_pick_succ", "ovmm_nav_orient_to_pick_succ", "ovmm_dist_to_pick_goal",
             "ovmm_nav_to_place_succ", "ovmm_nav_orient_to_place_succ", "ovmm_dist_to_place_goal",
             "ovmm_find_object_phase_success", "ovmm_find_recep_phase_success", "navmesh_collision")


def viewpoints(goals, pf):
    out = []
    for gi, g in enumerate(goals or []):
        for v in g.view_points:
            p = np.array(v.agent_state.position, float)
            s = np.array(pf.snap_point(p), float)
            ok = bool(np.isfinite(s).all() and np.linalg.norm(s - p) < 0.05)
            out.append({"goal": gi, "p": p, "snapped": s, "navigable": ok, "goal_pos": np.array(g.position, float)})
    return out


def metrics(env):
    m = env.get_metrics()
    return {k: (float(m[k]) if isinstance(m.get(k), (int, float, np.floating, bool)) else m.get(k)) for k in OVMM_KEYS if k in m}


def best_target(env, vps):
    """Navigable view point with the shortest geodesic distance from the robot."""
    best = None
    for v in vps:
        if not v["navigable"]:
            continue
        ok, _, geo = NAV.plan(env, v["snapped"])
        if ok and np.isfinite(geo) and (best is None or geo < best[1]):
            best = (v, geo)
    return best


with open(f"{OUT}/results_w{WK:02d}.jsonl", "a") as fout:
    for eid in eps:
        t0 = time.time()
        rec = {"episode_id": eid}
        try:
            HE.reset_to(env, eid)
            ep = env.current_episode
            pf = env.sim.pathfinder
            rob = HE.robot(env)
            start = np.array(rob.sim_obj.translation, float)
            snap = np.array(pf.snap_point(start), float)
            rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object_category": ep.object_category,
                        "goal_recep_category": ep.goal_recep_category,
                        "start_navigable": bool(pf.is_navigable(start)),
                        "start_snap_dist_m": float(np.linalg.norm(snap - start)) if np.isfinite(snap).all() else None,
                        "start_clearance_m": float(pf.distance_to_closest_obstacle(start, 2.0))})
            rec["start_fix"] = HE.fix_start(env)
            NAV.move_arms_to_tuck(env)
            c0, _ = NAV.scene_contacts(env)
            rec["start_collisions_after_tuck"] = [c["link"] for c in c0]
            obj_vps = viewpoints(ep.candidate_objects, pf)
            rec_vps = viewpoints(ep.candidate_goal_receps, pf)
            rec["viewpoints"] = {"object_total": len(obj_vps), "object_navigable": sum(v["navigable"] for v in obj_vps),
                                 "recep_total": len(rec_vps), "recep_navigable": sum(v["navigable"] for v in rec_vps)}
            for leg, vps in (("to_object", obj_vps), ("to_receptacle", rec_vps)):
                bt = best_target(env, vps)
                if bt is None:
                    rec[leg] = {"path_found": False, "reason": "no navigable, reachable OVMM view point for AlphaBot"}
                    break
                v, geo = bt
                gp = v["goal_pos"]
                st = NAV.follow(env, v["snapped"], face_xz=np.array([gp[0], gp[2]]), arrive_m=0.05)
                st["target_view_point"] = v["snapped"].round(3).tolist()
                st["ovmm_metrics"] = metrics(env)
                rec[leg] = st
        except Exception as e:
            import traceback
            rec["engineering_error"] = repr(e)
            rec["tb"] = traceback.format_exc()[-1500:]
        rec["wall_s"] = round(time.time() - t0, 1)
        fout.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        fout.flush()
        print("NAV", json.dumps({k: rec.get(k) for k in ("episode_id", "scene", "start_navigable", "engineering_error")}),
              {leg: {k: rec.get(leg, {}).get(k) for k in ("path_found", "arrived", "arrival_dist_m", "collision_steps", "steps")}
               for leg in ("to_object", "to_receptacle")}, flush=True)
env.close()
