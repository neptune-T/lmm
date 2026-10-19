"""Open-loop replay of recorded pa_eval.py episodes (no policy, no planner): reset_to(unit) + fix_start + the
trace's executed JointMobileAction18 rows, then OVMM's own funnel at the end. A success that reproduces here is a
genuine simulator episode (deterministic fresh-reset contract), independent of the policy server.
    REPLAY=unit1=trace1.npz,unit2=trace2.npz SPLIT=train OUT=... run_habitat.sh pa_replay.py
"""
import json, os, sys, time

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402

FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"), ("full", "ovmm_place_success"))
OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"))


def funnel():
    m = env.get_metrics(); out = {}
    for name, k in FUNNEL:
        v = m.get(k)
        if isinstance(v, dict):
            v = any(bool(x) for x in v.values())
        out[name] = bool(v) if v is not None else False
    return out


for item in [x for x in os.environ["REPLAY"].split(",") if x]:
    unit, trace = item.split("=", 1)
    t0 = time.time()
    J = np.load(trace)["joint18"]
    ep_id, rep = HE.parse_unit(unit)
    HE.reset_to(env, ep_id, rep)
    fix = HE.fix_start(env)
    n = 0
    for j in J:
        if env.episode_over:
            break
        HE.step18(env, j); n += 1
    rec = {"unit": unit, "trace": trace, "steps_recorded": int(len(J)), "steps_replayed": n, "start_fix": fix,
           "funnel": funnel(), "wall_s": round(time.time() - t0, 1)}
    with open(f"{OUT}/replay.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("REPLAY", json.dumps(rec), flush=True)
env.close()
