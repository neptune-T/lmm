#!/usr/bin/env python3
"""Root selection for the expert-recovery feasibility pilot (CPU, deterministic; no models).

Source: v2-B autonomous rollouts of the matched perturbation run, level "none" (exact expert NAV start, no imposed
perturbation). A root = (rollout, step r): r a multiple of H_EXEC (a policy query step), r >= 20, the robot
>= OFF_M off the unit's expert navigate_to_object path, > FAR_M from the expert stop point, the executed arm
targets of steps r-5..r-1 within TUCK_MAX of the NAV pose (arms tucked) and no false reach at or before r.
Categories (one root per rollout; first pick the category with the fewest candidates):
  pre_false_reach   triggering rollouts, r in [t* - 60, t* - 20], closest to t* - 30
  near_stall        no progress (distance to the stop point fell < 0.25 m over the 50 steps before r);
                    triggering rollouts only before t* - 60; earliest such r
  off_path_no_fr    rollouts without any false reach; r = the middle of their off-path far steps
Balanced over base episodes (round-robin) and train / held-out groups; N_PER per category.
    recovery_roots.py --run RUNS/v2/perturb/full --expert-pose stall_diag.json --out roots.json
"""
import argparse, glob, hashlib, json, os
from pathlib import Path

import numpy as np

H_EXEC, STOP_V, STOP_W, REACH_M, TUCK_MAX, FAR_M, OFF_M, PROG_WIN, PROG_M = 5, 0.03, 0.05, 0.15, 0.08, 2.0, 0.5, 50, 0.25
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def candidates(trace, src, ref):
    z = np.load(trace); pose, act = z["pose"], z["act"]
    md = np.load(Path(src) / "metadata.npz")
    nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
    exz = md["base_xz_yaw"][nav, :2]; stop = exz[-1]
    pdev = np.linalg.norm(pose[:, None, :2] - exz[None], axis=2).min(1)
    dstop = np.linalg.norm(pose[:, :2] - stop, axis=1)
    dev = np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))
    trig = None
    for t in range(0, len(act) - H_EXEC + 1, H_EXEC):
        if dstop[t] > FAR_M and np.abs(act[t:t + H_EXEC, 16]).mean() < STOP_V and np.abs(act[t:t + H_EXEC, 17]).mean() < STOP_W \
                and dev[t:t + H_EXEC].max() > REACH_M:
            trig = t; break
    ok = []
    for r in range(20, len(act), H_EXEC):
        if trig is not None and r >= trig:
            break
        if pdev[r] >= OFF_M and dstop[r] > FAR_M and dev[r - 5:r].max() <= TUCK_MAX:
            prog = dstop[max(0, r - PROG_WIN)] - dstop[r]
            ok.append({"r": r, "path_dev_m": round(float(pdev[r]), 3), "dist_to_stop_m": round(float(dstop[r]), 3),
                       "progress_last50_m": round(float(prog), 3)})
    return trig, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--n-per", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.run}/perturb_w*.jsonl")) for l in open(f)]
    none = sorted((r for r in rows if r["level"] == "none" and "steps" in r and os.path.exists(f"{a.run}/traces/{r['job']}.npz")),
                  key=lambda r: r["job"])
    pools = {"pre_false_reach": [], "near_stall": [], "off_path_no_fr": []}
    for r in none:
        tr = f"{a.run}/traces/{r['job']}.npz"
        trig, ok = candidates(tr, r["source"], ref)
        base = {"job": r["job"], "unit": r["unit"], "base_episode": r["unit"].split("#")[0], "group": r["group"], "source": r["source"],
                "t": r["t"], "seed": r["seed"], "trace": tr, "trace_sha256": hashlib.sha256(open(tr, "rb").read()).hexdigest(),
                "trigger_step": trig}
        if trig is not None:
            pre = [c for c in ok if trig - 60 <= c["r"] <= trig - 20]
            if pre:
                pools["pre_false_reach"].append({**base, **min(pre, key=lambda c: abs(c["r"] - (trig - 30)))})
            ns = [c for c in ok if c["r"] < trig - 60 and c["progress_last50_m"] < PROG_M]
            if ns:
                pools["near_stall"].append({**base, **ns[0]})
        else:
            ns = [c for c in ok if c["progress_last50_m"] < PROG_M]
            if ns:
                pools["near_stall"].append({**base, **ns[0]})
            if ok:
                pools["off_path_no_fr"].append({**base, **ok[len(ok) // 2]})
    picked, used, ep_count = [], set(), {}
    for cat in sorted(pools, key=lambda c: len(pools[c])):              # scarcest category first
        cand = sorted((c for c in pools[cat] if c["job"] not in used), key=lambda c: c["job"])
        groups = sorted({c["group"] for c in cand})
        n, gi = 0, 0
        while n < a.n_per and cand:                                       # alternate groups; least-used base episode
            g = groups[gi % len(groups)]; gi += 1
            cg = [c for c in cand if c["group"] == g]
            if not cg:
                if not any(c["group"] in groups for c in cand):
                    break
                continue
            c = min(cg, key=lambda c: (ep_count.get(c["base_episode"], 0), c["job"]))
            cand.remove(c); cand = [x for x in cand if x["job"] != c["job"]]
            used.add(c["job"]); ep_count[c["base_episode"]] = ep_count.get(c["base_episode"], 0) + 1
            picked.append({**c, "category": cat, "root_id": f"{cat}:{c['job']}:r{c['r']}"}); n += 1
    out = {"source_run": a.run, "candidates": {k: len(v) for k, v in pools.items()}, "roots": picked,
           "counts": {k: sum(p["category"] == k for p in picked) for k in pools},
           "groups": {g: sum(p["group"] == g for p in picked) for g in ("train_unit", "heldout")},
           "base_episodes": len({p["base_episode"] for p in picked})}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("candidates", "counts", "groups", "base_episodes")}))


if __name__ == "__main__":
    main()
