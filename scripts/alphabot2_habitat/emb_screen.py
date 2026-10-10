"""Deterministic AlphaBot 2 embodiment-feasibility screen (no rollouts, no expert outcomes).

For every geometrically feasible episode (feasibility_screen.py: aperture + reach band):
  * reset/nav reachability: a seeded OVMM reset for each seed-bank replica r < REPLICAS (OVMM asserts at
    reset when no goal is reachable from the sampled start on the AlphaBot navmesh); start pose recorded;
  * pick feasibility per arm (OVMMExpert.plan_pick, plan only): navigable standoff within the crouched
    arm's reach band, reachable geodesic, a grasp option within the EG2-4C2 aperture along the actual
    closing axis, grasp + pre-grasp IK with +-3 cm robustness;
  * place feasibility per arm (OVMMExpert.plan_place, plan only): a reachable standoff at a goal
    receptacle with place + pre-place IK, the object held as the planned grasp holds it.
Planning uses navmesh/geometry/IK only and runs from the first reachable replica's start.
Every DETERMINISM_EVERY-th reachable unit is reset twice: the start pose must be identical.

    SCREEN=.../screen_train.json REPLICAS=3 WORKER=0 NWORKERS=4 OUT=... run_habitat.sh emb_screen.py
"""
import json, os, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert, SIDES  # noqa: E402

OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
REPLICAS = int(os.environ.get("REPLICAS", "3"))
EVERY = int(os.environ.get("DETERMINISM_EVERY", "25"))
rows = [r for r in json.load(open(os.environ["SCREEN"])) if r["feasible"]]
scenes = sorted({r["scene"] for r in rows})[WK::NW]                    # shard by scene: one scene load each
todo = [r for r in sorted(rows, key=lambda r: (r["scene"], int(r["episode_id"]))) if r["scene"] in scenes]
fn = f"{OUT}/emb_w{WK:02d}.jsonl"
done = {json.loads(l)["episode_id"] for l in open(fn) if l.strip()} if os.path.exists(fn) else set()
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"))


def pose():
    rob = HE.robot(env)
    t = rob.sim_obj.translation
    return [float(t[0]), float(t[1]), float(t[2]), float(rob.base_rot)]


def plan(side, ex, obj):
    out = {"pick": False, "place": False}
    pk = ex.plan_pick(side, obj)
    out["pick_diag"] = ex.log.get("pick_diag", {}).get(side)
    if pk is None:
        return out
    out["pick"] = True
    out["pick_plan"] = {"mode": pk["grasp"]["mode"], "tilt": round(float(pk["grasp"]["tilt"]), 3),
                        "width_m": round(float(pk["grasp"]["width"]), 4), "geo_m": round(float(pk["geo"]), 3)}
    # the object as the planned grasp would hold it: its centre in gripper_base at grasp time
    centre, ext, R_obj, half = ex.object_geometry(obj)
    A = ex._arm_frame_at(pk["standoff"]["xyz"], pk["standoff"]["yaw"])
    o_arm = (np.linalg.inv(A) @ np.r_[centre, 1.0])[:3]
    g = pk["grasp"]
    T_gb = ex.pose(g["R"], g["pad"])
    ex.grasp_used = g
    ex.obj_in_gb = np.eye(4)
    ex.obj_in_gb[:3, 3] = (np.linalg.inv(T_gb) @ np.r_[o_arm, 1.0])[:3]
    pl = ex.plan_place(side, ext[1])
    out["place"] = pl is not None
    out["place_candidates_tried"] = ex.log.get("place_candidates_tried")
    if pl is not None:
        out["place_plan"] = {"recep": pl["recep"], "geo_m": round(float(pl["cost"]), 3)}
    return out


with open(fn, "a") as f:
    for k, r in enumerate(todo):
        eid = r["episode_id"]
        if eid in done:
            continue
        rec = {"episode_id": eid, "scene": r["scene"], "object": r.get("object"), "replicas": []}
        t0 = time.time()
        try:
            first = None
            for rep in range(REPLICAS):
                u = {"replica": rep, "seed": HE.episode_seed(eid, rep)}
                try:
                    HE.reset_to(env, eid, rep)
                    u["reachable"] = True
                    u["start_pose"] = pose()
                    if k % EVERY == 0:
                        HE.reset_to(env, eid, rep)
                        u["repeat_start_pose_equal"] = bool(np.allclose(pose(), u["start_pose"], atol=1e-6))
                    if first is None:
                        first = rep
                except AssertionError as e:
                    u["reachable"] = False
                    u["reason"] = str(e)[:120]
                rec["replicas"].append(u)
            rec["arms"] = {}
            if first is not None:
                HE.reset_to(env, eid, first)
                rec["start_fix"] = HE.fix_start(env)
                ex = OVMMExpert(env)
                obj, _ = ex.target_object()
                for side in SIDES:
                    rec["arms"][side] = plan(side, ex, obj)
            rec["feasible_arms"] = [s for s, a in rec["arms"].items() if a["pick"] and a["place"]]
        except Exception as e:
            rec["engineering_error"] = repr(e)[:300]
            rec["tb"] = traceback.format_exc()[-800:]
        rec["wall_s"] = round(time.time() - t0, 2)
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        f.flush()
print("###EMB_DONE###", WK, flush=True)
env.close()
