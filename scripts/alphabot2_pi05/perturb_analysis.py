#!/usr/bin/env python3
"""Analysis of perturb_eval.py rollouts (CPU, read-only). Per rollout, from the trace and the unit's expert
navigate_to_object path (stop point = its last pose); queries every H_EXEC steps; far = > FAR_M from the stop point.
  false_reach        a far query with stop (mean |v| < 0.03, |omega| < 0.05) and arm > T off the NAV pose
  onset_step         first such query (None if none); early = within EARLY_STEPS
  hazard             unified first-trigger statistic: events / eligible far queries up to the first trigger
  base_only_stall    far steps not moving with the arm within TUCK_MAX and commanded |v|,|omega| ~ 0
  stall_false_reach  far steps not moving with the arm > T (share of all far stall steps)
  progress_m         distance to the stop point at the start minus its minimum over the rollout
  reached_stop_1m    came within 1 m of the expert stop point
  recovery           after a false-reach onset: >= RECOVER_STEPS consecutive moving steps with the arm back
                     within TUCK_MAX, followed by >= 0.5 m more progress than at the onset
Per level: rates (Wilson 95 %), means; matched vs "none" on the same (state, seed): worse / better / tie, exact
sign test, bootstrap 95 % CI of the mean difference; per axis (pos, yaw): Cochran-Armitage trend test of
false_reach over (none, small, medium, large). T in 10 / 15 / 20 cm.
    perturb_analysis.py --run DIR --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, math, os
from pathlib import Path

import numpy as np

H_EXEC, STOP_V, STOP_W, MOVE_V, MOVE_W, DT = 5, 0.03, 0.05, 0.03, 0.05, 0.1
THRESHOLDS, TUCK_MAX, FAR_M, EARLY_STEPS, RECOVER_STEPS = (0.10, 0.15, 0.20), 0.08, 2.0, 100, 20
AXES = {"pos": ["none", "pos0.25", "pos0.5", "pos1.0"], "yaw": ["none", "yaw15", "yaw30", "yaw60"]}
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def rollout(trace, src, ref, T):
    z = np.load(trace); pose, act = z["pose"], z["act"]
    md = np.load(Path(src) / "metadata.npz")
    nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
    stop = md["base_xz_yaw"][nav[-1], :2]
    dstop = np.linalg.norm(pose[:, :2] - stop, axis=1)
    d = np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]); yaw = pose[:, 2]
    moving = (np.linalg.norm(d, axis=1) / DT >= MOVE_V) | (np.abs(np.angle(np.exp(1j * np.diff(yaw, prepend=yaw[:1])))) / DT >= MOVE_W)
    dev = np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))
    cmd0 = (np.abs(act[:, 16]) < STOP_V) & (np.abs(act[:, 17]) < STOP_W)
    far = dstop > FAR_M
    onset, eligible = None, 0
    for t in range(0, len(act) - H_EXEC + 1, H_EXEC):
        if not far[t]:
            continue
        c = act[t:t + H_EXEC]
        eligible += 1
        if np.abs(c[:, 16]).mean() < STOP_V and np.abs(c[:, 17]).mean() < STOP_W and dev[t:t + H_EXEC].max() > T:
            onset = t; break
    stall = far & ~moving
    rec = None
    if onset is not None:
        ok = (moving & (dev <= TUCK_MAX))[onset:]
        run = 0; resume = None
        for i, x in enumerate(ok):
            run = run + 1 if x else 0
            if run >= RECOVER_STEPS:
                resume = onset + i; break
        rec = bool(resume is not None and dstop[onset] - dstop[resume:].min() >= 0.5)
    return {"steps": len(act), "false_reach": onset is not None, "onset_step": onset,
            "early_false_reach": onset is not None and onset < EARLY_STEPS, "eligible": eligible, "events": int(onset is not None),
            "base_only_stall_steps": int((stall & (dev <= TUCK_MAX) & cmd0).sum()), "stall_steps": int(stall.sum()),
            "stall_false_reach_steps": int((stall & (dev > T)).sum()), "progress_m": float(dstop[0] - dstop.min()),
            "reached_stop_1m": bool(dstop.min() < 1.0), "start_dist_m": float(dstop[0]), "recovered": rec}


def wilson(k, n):
    if not n:
        return [None, None]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def sign_p(w, l):
    n = w + l
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n), 4) if n else None


def cochran_armitage(ks, ns):
    ks, ns = np.asarray(ks, float), np.asarray(ns, float); x = np.arange(len(ks), dtype=float)
    N, K = ns.sum(), ks.sum()
    if N == 0 or K in (0, N):
        return {"z": None, "p": None}
    p = K / N
    T = (x * (ks - ns * p)).sum()
    V = p * (1 - p) * ((ns * x * x).sum() - (ns * x).sum() ** 2 / N)
    zz = T / math.sqrt(V) if V > 0 else None
    return {"z": round(zz, 3) if zz is not None else None, "p": round(math.erfc(abs(zz) / math.sqrt(2)), 4) if zz is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with perturb_w*.jsonl and traces/")
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.run}/perturb_w*.jsonl")) for l in open(f)]
    rep = {"definition": __doc__, "rows": len(rows),
           "skipped": {k: sum(1 for r in rows if str(r.get("skipped", "")).startswith(k)) for k in ("replay_mismatch", '{"unsafe"')},
           "engineering_errors": sum(1 for r in rows if "engineering_error" in r),
           "replay_err_max": {"pos_m": max((r["replay"]["pos_err_m"] for r in rows if "replay" in r), default=None),
                              "yaw_deg": max((r["replay"]["yaw_err_deg"] for r in rows if "replay" in r), default=None)},
           "applied": {}, "by_threshold": {}}
    ok = [r for r in rows if "steps" in r and os.path.exists(f"{a.run}/traces/{r['job']}.npz")]
    for L in sorted({r["level"] for r in ok}):
        p = [r["perturbation"] for r in ok if r["level"] == L]
        rep["applied"][L] = {"n": len(p), "moved_m_med": round(float(np.median([x["moved_m"] for x in p])), 3),
                             "dyaw_deg_abs_med": round(float(np.median([abs(x["dyaw_deg"]) for x in p])), 2),
                             "minus_side": sum(x["sign"] < 0 for x in p)}
    rng = np.random.default_rng(0)
    for T in THRESHOLDS:
        R = {}
        for r in ok:
            m = rollout(f"{a.run}/traces/{r['job']}.npz", r["source"], ref, T)
            R[(r["unit"], r["t"], r["seed"], r["level"])] = {**m, "group": r["group"]}
        out = {"levels": {}, "matched_vs_none": {}, "trend": {}}
        for L in sorted({k[3] for k in R}):
            v = [m for k, m in R.items() if k[3] == L]
            fr = sum(m["false_reach"] for m in v); n = len(v)
            onsets = [m for m in v if m["false_reach"]]
            out["levels"][L] = {"n": n, "false_reach": fr, "rate": round(fr / n, 3) if n else None, "rate_ci95": wilson(fr, n),
                                "early_false_reach": sum(m["early_false_reach"] for m in v),
                                "hazard": round(sum(m["events"] for m in v) / max(1, sum(m["eligible"] for m in v)), 5),
                                "eligible_queries": sum(m["eligible"] for m in v),
                                "base_only_stall_steps_mean": round(float(np.mean([m["base_only_stall_steps"] for m in v])), 1),
                                "stall_steps_mean": round(float(np.mean([m["stall_steps"] for m in v])), 1),
                                "stall_share_false_reach": round(sum(m["stall_false_reach_steps"] for m in v) / max(1, sum(m["stall_steps"] for m in v)), 3),
                                "progress_m_mean": round(float(np.mean([m["progress_m"] for m in v])), 2),
                                "reached_stop_1m": sum(m["reached_stop_1m"] for m in v),
                                "recovered": sum(bool(m["recovered"]) for m in onsets), "recovery_of": len(onsets),
                                "by_group": {g: {"n": sum(m["group"] == g for m in v), "false_reach": sum(m["false_reach"] for m in v if m["group"] == g)}
                                             for g in ("train_unit", "heldout")}}
        for L in sorted({k[3] for k in R} - {"none"}):
            pairs = [(R[k], R[(k[0], k[1], k[2], "none")]) for k in R if k[3] == L and (k[0], k[1], k[2], "none") in R]
            res = {"n_pairs": len(pairs)}
            w = sum(x["false_reach"] and not y["false_reach"] for x, y in pairs); b = sum(y["false_reach"] and not x["false_reach"] for x, y in pairs)
            res["false_reach"] = {"perturbed_only": w, "none_only": b, "sign_test_p": sign_p(w, b)}
            for k in ("progress_m", "stall_steps", "base_only_stall_steps"):
                dd = np.array([x[k] - y[k] for x, y in pairs], float)
                boot = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)] if len(dd) else [0]
                res[k] = {"mean_diff": round(float(dd.mean()), 3) if len(dd) else None,
                          "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}
            out["matched_vs_none"][L] = res
        for ax, Ls in AXES.items():
            ks = [out["levels"].get(L, {}).get("false_reach", 0) for L in Ls]; ns = [out["levels"].get(L, {}).get("n", 0) for L in Ls]
            out["trend"][ax] = {"levels": Ls, "false_reach": ks, "n": ns, **cochran_armitage(ks, ns)}
        rep["by_threshold"][f"{T:.2f}"] = out
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("rows", "skipped", "engineering_errors", "replay_err_max", "applied")}))
    for T, o in rep["by_threshold"].items():
        print(f"== T={T}")
        for L, x in o["levels"].items():
            print(f"   {L:8s} n={x['n']:3d} false_reach {x['false_reach']:3d} ({x['rate']}, {x['rate_ci95']}) early {x['early_false_reach']:3d} hazard {x['hazard']} "
                  f"stall_mean {x['stall_steps_mean']} base_only {x['base_only_stall_steps_mean']} FRshare {x['stall_share_false_reach']} "
                  f"progress {x['progress_m_mean']} reach1m {x['reached_stop_1m']} recovered {x['recovered']}/{x['recovery_of']} {x['by_group']}")
        for ax, t in o["trend"].items():
            print(f"   trend {ax}: {t['false_reach']}/{t['n']} z {t['z']} p {t['p']}")
        for L, m in o["matched_vs_none"].items():
            print(f"   matched {L:8s} {json.dumps(m)}")


if __name__ == "__main__":
    main()
