"""Multi-process reset reproducibility audit (A->B->A) for AlphaBot 2 OVMM units.

Each worker process resets the audit units in its own seeded order (so every unit is reset after
different predecessors, in different processes) and, for A->B->A pairs, runs the full expert on B
between two resets of A. Per (unit, worker, occasion): reset outcome (OVMM's reachability assert) and
the sampled start pose. A unit whose outcome or start pose is not identical everywhere is
RESET_NONDETERMINISTIC and is excluded from the AlphaBot-feasible list (never enters the dataset).

    UNITS=a#0,b#0,... PAIRS=a#0:b#0,... WORKER=k NWORKERS=n OUT=... run_habitat.sh reset_audit.py
    python3 reset_audit.py --summarize OUT       (stdlib; writes OUT/reset_audit.json)
"""
import json, os, random, sys


def summarize(out):
    rows = []
    for i in range(64):
        f = f"{out}/audit_w{i:02d}.jsonl"
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["unit"], []).append(r)
    bad = {}
    for u, rs in by.items():
        oks = {r["ok"] for r in rs}
        poses = {tuple(round(x, 5) for x in r["start_pose"]) for r in rs if r["ok"]}
        if len(oks) > 1 or len(poses) > 1:
            bad[u] = {"outcomes": sorted(map(str, oks)), "distinct_start_poses": len(poses), "observations": len(rs),
                      "contexts": sorted({r["context"] for r in rs})}
    res = {"units": len(by), "observations": len(rows), "workers": len({r["worker"] for r in rows}),
           "reset_nondeterministic": sorted(bad), "detail": bad}
    json.dump(res, open(f"{out}/reset_audit.json", "w"), indent=1)
    print("###RESET_AUDIT###" + json.dumps({k: res[k] for k in ("units", "observations", "workers", "reset_nondeterministic")}))


if __name__ == "__main__" and "--summarize" in sys.argv:
    summarize(sys.argv[-1])
    sys.exit(0)

import numpy as np  # noqa: E402

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
units = [u for u in os.environ["UNITS"].split(",") if u]
pairs = [p.split(":") for p in os.environ.get("PAIRS", "").split(",") if p]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"))
fn = f"{OUT}/audit_w{WK:02d}.jsonl"


def reset(unit, context):
    e, r = HE.parse_unit(unit)
    rec = {"unit": unit, "worker": WK, "context": context}
    try:
        HE.reset_to(env, e, r)
        rob = HE.robot(env)
        t = rob.sim_obj.translation
        rec.update({"ok": True, "start_pose": [float(t[0]), float(t[1]), float(t[2]), float(rob.base_rot)]})
    except AssertionError as ex:
        rec.update({"ok": False, "start_pose": None, "reason": str(ex)[:120]})
    with open(fn, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


order = units[:]
random.Random(1000 + WK).shuffle(order)
for u in order:
    reset(u, f"order_w{WK}")
for a, b in pairs[WK::NW]:
    reset(a, "A_before")
    e, r = HE.parse_unit(b)
    try:
        HE.reset_to(env, e, r)
        HE.fix_start(env)
        OVMMExpert(env).run()
    except Exception:
        pass
    reset(a, f"A_after_expert_on_{b}")
    reset(a, "A_again")
print("###AUDIT_WORKER_DONE###", WK, flush=True)
env.close()
