"""Geometric feasibility probe for perturb_eval.py (no policy): for every planned expert NAV state, replay it
exactly as perturb_eval.py does, then test every position magnitude on BOTH sides and every yaw magnitude in BOTH
directions (navigable, same navmesh island, robot-scene contacts after the teleport; the state is restored after
each test). Also the clearance (distance to the closest obstacle) at the state and at each lateral candidate.
Gives each level's feasible sample size before any rollout and shows whether one side is systematically blocked.
    JOBS=jobs.json OUT=... GPU=.. WORKER=.. NWORKERS=.. run_habitat.sh perturb_probe.py
"""
import json, math, os, sys

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
import magnum as mn  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402

OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
plan = json.load(open(os.environ["JOBS"]))
states = sorted({(j["unit"], j["t"], j["source"]) for j in plan["jobs"]})[WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=plan.get("split", "train"))
fn = f"{OUT}/probe_w{WK:02d}.jsonl"
for unit, t, src in states:
    rec = {"unit": unit, "t": t}
    try:
        ep_id, rep = HE.parse_unit(unit)
        low, md = np.load(f"{src}/lowdim.npz"), np.load(f"{src}/metadata.npz")
        HE.reset_to(env, ep_id, rep); HE.fix_start(env)
        for a in low["joint18"][:t]:
            HE.step18(env, a)
        rob = HE.robot(env); pf = env.sim.pathfinder
        xz, yaw, _ = NAV.pose2d(env)
        rec["replay_pos_err_m"] = float(np.linalg.norm(xz - md["base_xz_yaw"][t, :2]))
        T0 = rob.sim_obj.transformation; R0, p0 = T0.rotation(), T0.translation
        island0 = pf.get_island(p0)
        rec["clearance_m"] = float(pf.distance_to_closest_obstacle(p0, 3.0))
        rec["root_navigable"] = bool(pf.is_navigable(p0)); rec["root_y"] = float(p0[1]); rec["snap_y"] = float(pf.snap_point(p0)[1])
        left = np.array([-math.sin(yaw), -math.cos(yaw)])
        res = {}
        for mag in (0.25, 0.5, 1.0):
            for sign in (1.0, -1.0):
                off = sign * mag * left
                p = mn.Vector3(p0[0] + off[0], p0[1], p0[2] + off[1])
                nav = bool(pf.is_navigable(p)); same = nav and pf.get_island(p) == island0
                coll = None
                if same:
                    rob.sim_obj.transformation = mn.Matrix4.from_(R0, p); rob.update()
                    coll = len(NAV.scene_contacts(env)[0])
                    rob.sim_obj.transformation = mn.Matrix4.from_(R0, p0); rob.update()
                res[f"pos{mag:g}{'+' if sign > 0 else '-'}"] = {"navigable": nav, "same_island": same, "contacts": coll,
                                                               "clearance_m": float(pf.distance_to_closest_obstacle(p, 3.0)) if nav else None}
        for mag in (15.0, 30.0, 60.0):
            for sign in (1.0, -1.0):
                R = mn.Matrix4.rotation_y(mn.Rad(sign * math.radians(mag))).rotation() @ R0
                rob.sim_obj.transformation = mn.Matrix4.from_(R, p0); rob.update()
                coll = len(NAV.scene_contacts(env)[0])
                rob.sim_obj.transformation = mn.Matrix4.from_(R0, p0); rob.update()
                res[f"yaw{mag:g}{'+' if sign > 0 else '-'}"] = {"contacts": coll}
        rec["tests"] = res
    except Exception as e:
        rec["error"] = repr(e)[:300]
    with open(fn, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("PROBE", json.dumps({k: rec.get(k) for k in ("unit", "t", "replay_pos_err_m", "clearance_m", "error")}), flush=True)
env.close()
