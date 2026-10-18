#!/usr/bin/env python3
"""DAgger recovery SFT pilot: paired per-(root, seed) comparison of policy continuations (CPU, read-only).

Arms: CONTROL and RECOVERY (fine-tuned from v2-B best, `last` checkpoint) and the original v2-B baseline from the
recovery feasibility run. Same 36 roots, same two noise seeds (policy_orig / policy_alt), 600 steps each.
Outcomes per continuation (recovery_analysis.policy_metrics): approach (target <= 1 m), false reach (> 2 m from the
expert stop point), recovery success = approach with no false reach before the first approach step.
PRIMARY = held-out roots (pre-registered, docs/v2/nav_decision_preregistration.md section 4); training-unit roots
(in-distribution of the recovery data) are reported separately and never pooled with them.
    pilot_compare.py --arm control=DIR --arm recovery=DIR --baseline B=DIR --roots roots.json --expert-pose POSE --out OUT
"""
import argparse, glob, json, math, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recovery_analysis import policy_metrics  # noqa: E402


def sign_p(w, l):
    n = w + l
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n), 4) if n else None


def load(run, roots, ref):
    out = {}
    for f in sorted(glob.glob(f"{run}/recovery_w*.jsonl")):
        for l in open(f):
            rec = json.loads(l)
            if rec["mode"] not in ("policy_orig", "policy_alt") or "trace_out" not in rec:
                continue
            r = roots[rec["root_id"]]
            md = np.load(Path(r["source"]) / "metadata.npz")
            nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
            m = policy_metrics(rec, np.load(rec["trace_out"], allow_pickle=True), ref, md["base_xz_yaw"][nav[-1], :2])
            m["recovery_success"] = bool(m["approach"] and (not m["false_reach"] or m["false_reach_step"] > m["steps_to_approach"]))
            m["policy_ckpt"] = rec.get("policy_ckpt")
            out[(rec["root_id"], rec["mode"])] = m
    return out


def summary(R, keys):
    v = [R[k] for k in keys if k in R]
    return {"n": len(v), "approach": sum(m["approach"] for m in v), "false_reach": sum(m["false_reach"] for m in v),
            "recovery_success": sum(m["recovery_success"] for m in v), "find": sum(bool(m["funnel_end"] and m["funnel_end"].get("find")) for m in v),
            "contact_continuations": sum(m["contact_steps"] > 0 for m in v),
            "min_dist_median_m": round(float(np.median([m["min_dist_to_target_m"] for m in v])), 2) if v else None}


def paired(X, Y, keys, metric, higher_better=True):
    ks = [k for k in keys if k in X and k in Y]
    w = sum(bool(X[k][metric]) and not bool(Y[k][metric]) for k in ks); l = sum(bool(Y[k][metric]) and not bool(X[k][metric]) for k in ks)
    if not higher_better:
        w, l = l, w
    return {"n_pairs": len(ks), "x_better": w, "y_better": l, "sign_test_p": sign_p(w, l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--baseline", required=True, help="NAME=DIR (the feasibility run: v2-B best)")
    ap.add_argument("--roots", required=True)
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    roots = {r["root_id"]: r for r in json.load(open(a.roots))["roots"]}
    runs = dict(x.split("=", 1) for x in a.arm); bname, bdir = a.baseline.split("=", 1); runs[bname] = bdir
    R = {n: load(d, roots, ref) for n, d in runs.items()}
    rep = {"definition": __doc__, "ckpts": {n: sorted({m["policy_ckpt"] for m in r.values()}, key=str) for n, r in R.items()}, "groups": {}}
    for g in ("heldout", "train_unit"):
        keys = [(rid, m) for rid, r in roots.items() if r["group"] == g for m in ("policy_orig", "policy_alt")]
        G = {"roots": sum(1 for r in roots.values() if r["group"] == g), "summary": {n: summary(r, keys) for n, r in R.items()}, "paired": {}}
        names = list(runs)
        for x, y in (("recovery", "control"), ("recovery", bname), ("control", bname)):
            if x in R and y in R:
                G["paired"][f"{x}_vs_{y}"] = {m: paired(R[x], R[y], keys, m, hb) for m, hb in
                                              (("approach", True), ("recovery_success", True), ("false_reach", False))}
        rep["groups"][g] = G
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"ckpts": rep["ckpts"]}))
    for g, G in rep["groups"].items():
        print(f"== {g} ({G['roots']} roots x 2 seeds)")
        for n, s in G["summary"].items():
            print(f"   {n:9s} {json.dumps(s)}")
        for k, v in G["paired"].items():
            print(f"   paired {k}: {json.dumps(v)}")


if __name__ == "__main__":
    main()
