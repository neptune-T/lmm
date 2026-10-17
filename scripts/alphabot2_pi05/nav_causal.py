#!/usr/bin/env python3
"""NAV causal control + paired A/B comparison (CPU, read-only: saved teacher-forced probe outputs, eval traces,
eval result rows).

1. Unified early-reach onset statistic (teacher-forced and closed loop identically). Queries in time order at the
   same cadence (every H_EXEC = 5 steps / frames). An episode is EXPOSED from its first far-NAV query up to and
   including its first early-reach query (stop and arm offset > T); later queries are not eligible. Then
     first_trigger_rate = episodes with a first trigger / eligible queries      (events per exposure query)
     episodes_triggered = episodes with a first trigger / episodes
   by distance (the event and its exposure go to the distance bin of each query) and, closed loop on the training
   units, by the robot's path state AT EACH QUERY (deviation from the same unit's expert navigate_to_object path:
   on < 0.5 m, mid 0.5-1 m, off >= 1 m), so on-path and off-path hazards have their own denominators.
   Far NAV: > FAR_M from the expert's stop point (teacher-forced: expert pose; closed loop on training units: robot
   pose vs the same unit's expert stop point), closed loop after TUCK_SKIP steps.
2. Drift ordering: for each closed-loop training-unit episode with a first trigger, the first query with path
   deviation >= 1 m (drift onset) vs the trigger: drift >= 50 steps before / within 50 steps before (ambiguous) /
   no drift before the trigger.
3. Paired per-episode NAV comparison (same unit and noise seed): stall fraction, closest distance, reached 1 m,
   first early reach (15 cm), path length from traces; find from the eval rows (both seeds). Per pair: n, X better /
   worse / tie, median difference, mean difference with a bootstrap 95 % CI, exact two-sided sign-test p.
    nav_causal.py --runs /workspace/runtime/runs/v2 --models A B BASE --expert-pose POSE.json --out OUT.json
"""
import argparse, glob, json, math, os
from pathlib import Path

import numpy as np

H_EXEC, STOP_V, STOP_W = 5, 0.03, 0.05
THRESHOLDS = (0.10, 0.15, 0.20)
FAR_M, TUCK_SKIP, DRIFT_M, LEAD_STEPS = 2.0, 20, 1.0, 50
DIST_BINS = [(2, 4), (4, 8), (8, 99)]
PATH_STATES = (("on_lt_0.5m", 0, 0.5), ("mid_0.5_1m", 0.5, 1.0), ("off_ge_1m", 1.0, 99))
LXYZ, RXYZ = slice(0, 3), slice(7, 10)
MOVE_V, MOVE_W = 0.03, 0.05


def chunk_dev(c, ref):
    return max(float(np.linalg.norm(c[:, LXYZ] - ref["left"], axis=1).max()), float(np.linalg.norm(c[:, RXYZ] - ref["right"], axis=1).max()))


def is_stop(c):
    return bool(np.abs(c[:, 16]).mean() < STOP_V and np.abs(c[:, 17]).mean() < STOP_W)


def expert_nav(collect, unit):
    ep, rep = unit.split("#")
    d = Path(collect) / "episodes" / f"{ep}_r{rep}"
    if not (d / "metadata.npz").exists():
        return None
    md = np.load(d / "metadata.npz")
    nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
    return md["base_xz_yaw"][nav, :2] if nav else None


# ---------------------------------------------------------------- episode query sequences
def tf_episodes(path):
    """teacher-forced: {episode: [query...]} far-NAV queries in time order"""
    if not os.path.exists(path):
        return None
    eps = {}
    for q in json.load(open(path))["queries_detail"]:
        if q["dist"] > FAR_M:
            eps.setdefault(q["unit"], []).append({"t": q["t"], "dist": q["dist"], "stop": bool(q["stop"]), "dev": float(q["dev"]), "path_dev": None})
    for k in eps:
        eps[k].sort(key=lambda q: q["t"])
    return eps


def trace_files(d):
    return sorted(glob.glob(os.path.join(d, "*.npz"))) if d and os.path.isdir(d) else []


def unit_of(f):
    return Path(f).stem.replace("trace_", "").replace("_r", "#")


def cl_episodes(d, ref, collect):
    """closed loop on training units: far-NAV queries (vs the unit's expert stop point) with path deviation"""
    if not trace_files(d):
        return None
    eps = {}
    for f in trace_files(d):
        unit, path = unit_of(f), expert_nav(collect, unit_of(f))
        if path is None:
            continue
        z = np.load(f, allow_pickle=True); pose, act = z["pose"], z["act"]
        qs = []
        for t in range(TUCK_SKIP - TUCK_SKIP % H_EXEC, len(act) - H_EXEC + 1, H_EXEC):
            dist = float(np.linalg.norm(pose[t, :2] - path[-1]))
            if dist <= FAR_M:
                continue
            c = act[t:t + H_EXEC]
            qs.append({"t": t, "dist": dist, "stop": is_stop(c), "dev": chunk_dev(c, ref),
                       "path_dev": float(np.min(np.linalg.norm(path - pose[t, :2], axis=1)))})
        eps[unit] = qs
    return eps


def onset(eps, T):
    """unified first-trigger statistic"""
    n_ep = len(eps); events = exposure = 0
    by_dist = {f"{lo}-{hi}m": [0, 0] for lo, hi in DIST_BINS}
    by_path = {s: [0, 0] for s, _, _ in PATH_STATES}
    firsts = {}
    for u, qs in eps.items():
        for q in qs:
            trig = q["stop"] and q["dev"] > T
            exposure += 1
            for lo, hi in DIST_BINS:
                if lo <= q["dist"] < hi:
                    by_dist[f"{lo}-{hi}m"][1] += 1; by_dist[f"{lo}-{hi}m"][0] += trig
            if q["path_dev"] is not None:
                for s, lo, hi in PATH_STATES:
                    if lo <= q["path_dev"] < hi:
                        by_path[s][1] += 1; by_path[s][0] += trig
            if trig:
                events += 1; firsts[u] = q
                break
    rate = lambda e, x: round(e / x, 5) if x else None
    return {"episodes": n_ep, "episodes_triggered": len(firsts), "eligible_queries": exposure,
            "first_trigger_rate": rate(events, exposure),
            "by_dist": {k: {"events": e, "eligible": x, "rate": rate(e, x)} for k, (e, x) in by_dist.items()},
            "by_path_state": {k: {"events": e, "eligible": x, "rate": rate(e, x)} for k, (e, x) in by_path.items()} if any(v[1] for v in by_path.values()) else None}, firsts


def drift_order(eps, firsts):
    out = {"drift_ge_50_steps_before": 0, "drift_within_50_steps_before": 0, "no_drift_before": 0, "episodes": len(firsts), "detail": {}}
    for u, q0 in firsts.items():
        drift = next((q["t"] for q in eps[u] if q["path_dev"] is not None and q["path_dev"] >= DRIFT_M and q["t"] <= q0["t"]), None)
        if drift is None:
            k = "no_drift_before"
        elif q0["t"] - drift >= LEAD_STEPS:
            k = "drift_ge_50_steps_before"
        else:
            k = "drift_within_50_steps_before"
        out[k] += 1
        out["detail"][u] = {"trigger_t": q0["t"], "drift_onset_t": drift, "path_dev_at_trigger": round(q0["path_dev"], 2), "dist_at_trigger": round(q0["dist"], 2)}
    return out


# ---------------------------------------------------------------- paired per-episode NAV
def episode_nav(f, ref):
    z = np.load(f, allow_pickle=True); pose, act, dist = z["pose"], z["act"], z["dist"]
    d = np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]); yaw = pose[:, 2]
    moving = (np.linalg.norm(d, axis=1) / 0.1 >= MOVE_V) | (np.abs(np.angle(np.exp(1j * np.diff(yaw, prepend=yaw[:1])))) / 0.1 >= MOVE_W)
    first = None
    for t in range(TUCK_SKIP - TUCK_SKIP % H_EXEC, len(act) - H_EXEC + 1, H_EXEC):
        c = act[t:t + H_EXEC]
        if dist[t] > FAR_M and is_stop(c) and chunk_dev(c, ref) > 0.15:
            first = t; break
    return {"stall_frac": float((~moving[TUCK_SKIP:]).mean()), "min_dist": float(dist.min()), "reached_1m": float(dist.min() < 1.0),
            "path_m": float(np.linalg.norm(d, axis=1).sum()), "early_reach": float(first is not None),
            "early_reach_step": float(first) if first is not None else float(len(act))}


def find_rows(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if "funnel" in r and "unit" in r:
                rows[r["unit"]] = float(r["funnel"]["find"])
    return rows


def sign_p(w, l):
    n = w + l
    if n == 0:
        return None
    k = min(w, l)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def paired(x, y, lower_better, rng):
    x, y = np.asarray(x), np.asarray(y); diff = x - y
    better = (diff < 0) if lower_better else (diff > 0); worse = (diff > 0) if lower_better else (diff < 0)
    w, l = int(better.sum()), int(worse.sum())
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)] if len(diff) else [0]
    return {"n": int(len(diff)), "better": w, "worse": l, "tie": int(len(diff) - w - l), "median_diff": round(float(np.median(diff)), 4) if len(diff) else None,
            "mean_diff": round(float(diff.mean()), 4) if len(diff) else None,
            "mean_diff_ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)], "sign_test_p": sign_p(w, l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="eval suffixes: A B BASE")
    ap.add_argument("--trigger-names", nargs="+", default=None, help="trigger dir per model (default: model name; BASE -> sanity_v1)")
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--collect", default="/workspace/runtime/runs/collect_clean_v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    tnames = a.trigger_names or [("sanity_v1" if m == "BASE" else m) for m in a.models]
    rep = {"definition": __doc__, "thresholds_m": THRESHOLDS, "onset": {}, "drift_order": {}, "paired_nav": {}, "per_episode": {}}
    rng = np.random.default_rng(0)
    for m, tn in zip(a.models, tnames):
        conds = {"tf_heldout_val": tf_episodes(f"{a.runs}/trigger/{tn}/heldout_val.json"),
                 "tf_train_units": tf_episodes(f"{a.runs}/trigger/{tn}/train_units.json"),
                 "cl_train_units": cl_episodes(f"{a.runs}/eval_{m}/traces_train_cl", ref, a.collect)}
        rep["onset"][m], rep["drift_order"][m] = {}, {}
        for c, eps in conds.items():
            if eps is None:
                rep["onset"][m][c] = "missing"; continue
            rep["onset"][m][c] = {}
            for T in THRESHOLDS:
                st, firsts = onset(eps, T)
                rep["onset"][m][c][f"{T:.2f}"] = st
                if c == "cl_train_units":
                    rep["drift_order"][m][f"{T:.2f}"] = drift_order(eps, firsts)
        # per-episode NAV for the paired comparison (traces: train set + OVMM-val seed 0; find: all eval rows)
        pe = {}
        for split, tdir in (("train", "traces_train_cl"), ("val0", "traces_val_pass0")):
            for f in trace_files(f"{a.runs}/eval_{m}/{tdir}"):
                pe[f"{split}:{unit_of(f)}"] = episode_nav(f, ref)
        for split, sub in (("train", "train_cl"), ("val0", "val_pass0"), ("val1", "val_pass1")):
            for u, fnd in find_rows(f"{a.runs}/eval_{m}/{sub}").items():
                pe.setdefault(f"{split}:{u}", {})["find"] = fnd
        rep["per_episode"][m] = pe
        for c, x in rep["onset"][m].items():
            if x != "missing":
                print(f"{m:5s} {c:16s} " + " | ".join(f"{T}: {s['episodes_triggered']}/{s['episodes']} eps, {s['eligible_queries']} elig, rate {s['first_trigger_rate']}" for T, s in x.items()), flush=True)
        if rep["drift_order"][m]:
            print(f"{m:5s} drift_order@0.15 " + json.dumps({k: v for k, v in rep["drift_order"][m]["0.15"].items() if k != "detail"}), flush=True)
    metrics = (("stall_frac", True), ("min_dist", True), ("reached_1m", False), ("early_reach", True), ("early_reach_step", False), ("path_m", False), ("find", False))
    pairs = [(x, y) for x, y in (("B", "A"), ("B", "BASE"), ("A", "BASE")) if x in a.models and y in a.models]
    for x, y in pairs:                                               # "better" = x better than y
        if True:
            px, py = rep["per_episode"][x], rep["per_episode"][y]
            res = {}
            for k, lower in metrics:
                keys = sorted(u for u in px if u in py and k in px[u] and k in py[u])
                if keys:
                    res[k] = paired([px[u][k] for u in keys], [py[u][k] for u in keys], lower, rng)
            rep["paired_nav"][f"{x}_vs_{y}"] = res
            print(f"paired {x} vs {y}: " + json.dumps({k: {kk: v[kk] for kk in ('n', 'better', 'worse', 'tie', 'median_diff', 'mean_diff_ci95', 'sign_test_p')} for k, v in res.items()}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
