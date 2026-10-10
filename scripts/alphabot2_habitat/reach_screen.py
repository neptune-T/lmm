"""Navmesh reachability screen for AlphaBot 2: a normal (per-episode seeded) OVMM reset of every
geometrically feasible episode; OVMM asserts at reset when no goal is reachable from the start on the
AlphaBot navmesh (radius 0.38 m). Records the outcome, the sampled start pose and the start fix.
Every DETERMINISM_EVERY-th episode is reset twice: the start pose must be identical.

    SCREEN=.../screen_train.json SPLIT=train WORKER=0 NWORKERS=4 OUT=... run_habitat.sh reach_screen.py
"""
import json, os, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402

OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
EVERY = int(os.environ.get("DETERMINISM_EVERY", "25"))
rows = [r for r in json.load(open(os.environ["SCREEN"])) if r["feasible"]]
scenes = sorted({r["scene"] for r in rows})[WK::NW]                    # shard by scene: one scene load each
todo = [r for r in sorted(rows, key=lambda r: (r["scene"], int(r["episode_id"]))) if r["scene"] in scenes]
done = set()
fn = f"{OUT}/reach_w{WK:02d}.jsonl"
if os.path.exists(fn):
    done = {json.loads(l)["episode_id"] for l in open(fn) if l.strip()}
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"))


def pose():
    rob = HE.robot(env)
    t = rob.sim_obj.translation
    return [float(t[0]), float(t[1]), float(t[2]), float(rob.base_rot)]


with open(fn, "a") as f:
    for k, r in enumerate(todo):
        if r["episode_id"] in done:
            continue
        rec = {"episode_id": r["episode_id"], "scene": r["scene"], "seed": HE.episode_seed(r["episode_id"])}
        t0 = time.time()
        try:
            HE.reset_to(env, r["episode_id"])
            rec["reachable"] = True
            rec["start_pose"] = pose()
            if k % EVERY == 0:
                HE.reset_to(env, r["episode_id"])
                rec["repeat_start_pose_equal"] = bool(np.allclose(pose(), rec["start_pose"], atol=1e-6))
            rec["start_fix"] = HE.fix_start(env)
        except AssertionError as e:
            rec["reachable"] = False
            rec["reason"] = str(e)[:160]
        except Exception as e:
            rec["engineering_error"] = repr(e)[:300]
            rec["tb"] = traceback.format_exc()[-800:]
        rec["wall_s"] = round(time.time() - t0, 2)
        f.write(json.dumps(rec) + "\n")
        f.flush()
print("###REACH_DONE###", WK, flush=True)
env.close()
