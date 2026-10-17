#!/usr/bin/env python3
"""Offline (CPU) re-scoring of the NAV early-reach probe at several arm thresholds, from saved outputs only:
teacher-forced expert_trigger.py results (raw per-query arm offset "dev") and closed-loop eval traces.

Per query (first H_EXEC steps of a predicted / executed chunk), dev = max over both arms and the H_EXEC steps of
|commanded arm xyz - expert NAV pose|, stop = mean |v| < 0.03 and mean |omega| < 0.05:
  early_reach      stop and dev > T                    (false NAV -> REACH transition)
  reach_moving     not stop and dev > T
  pure_stop        stop and dev <= 0.01                (base stop, arm at the NAV pose)
  stop_tuck_like   stop and 0.01 < dev <= 0.08         (expert tuck range <= 0.06)
  intermediate     0.08 < dev <= T                     (between tuck and the reach threshold)
Closed loop, a stuck robot re-triggers on every later query, so the per-query rate mostly measures how long it
stays stuck. The comparable quantity to a teacher-forced query (always taken from the expert's driving state) is
the ONSET HAZARD: P(early_reach at query t | no early_reach at query t-1); also where each unit's first far early
reach happens (on / off the expert path).
Only far NAV states are compared:
  expert   navigate_to_object frames > FAR_M from the expert's stop point (end of navigate_to_object)
  cl_train closed-loop queries on the training-set eval units after TUCK_SKIP steps, > FAR_M from the SAME expert
           stop point (from the robot pose); split on-path (< 0.5 m from the expert path) / off-path (>= 1 m)
  cl_val   OVMM-val eval units (no expert path): > FAR_M_VAL from the target object, after TUCK_SKIP steps
    rescore_trigger.py --model A=TRIGGER_DIR:TRACES_TRAIN:TRACES_VAL ... --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

H_EXEC, STOP_V, STOP_W = 5, 0.03, 0.05
THRESHOLDS = (0.10, 0.15, 0.20)
NAV_EPS, TUCK_MAX = 0.01, 0.08
FAR_M, FAR_M_VAL, TUCK_SKIP = 2.0, 2.5, 20
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def expert_nav(collect, unit):
    ep, rep = unit.split("#")
    d = Path(collect) / "episodes" / f"{ep}_r{rep}"
    if not (d / "metadata.npz").exists():
        return None
    md = np.load(d / "metadata.npz")
    nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
    return md["base_xz_yaw"][nav, :2] if nav else None


def cats(qs, T):
    n = len(qs)
    if not n:
        return {"queries": 0}
    st = np.array([q["stop"] for q in qs]); dv = np.array([q["dev"] for q in qs])
    trig = st & (dv > T)
    if "prev_stop" in qs[0]:                                         # closed loop: entries from a non-triggered query
        fresh = ~np.array([q["prev_stop"] and q["prev_dev"] > T for q in qs])
        hazard, n_fresh = (round(float(trig[fresh].mean()), 4) if fresh.any() else None), int(fresh.sum())
    else:                                                            # teacher forced: every query is an expert state
        hazard, n_fresh = round(float(trig.mean()), 4), n
    units = {}
    for q, e in zip(qs, st & (dv > T)):
        units[q["unit"]] = units.get(q["unit"], False) or bool(e)
    r = lambda m: round(float(np.mean(m)), 4)
    return {"queries": n, "units": len(units), "early_reach": r(st & (dv > T)), "reach_moving": r(~st & (dv > T)),
            "pure_stop": r(st & (dv <= NAV_EPS)), "stop_tuck_like": r(st & (dv > NAV_EPS) & (dv <= TUCK_MAX)),
            "intermediate": r((dv > TUCK_MAX) & (dv <= T)), "any_stop": r(st),
            "units_with_early_reach": sum(units.values()), "early_reach_count": int((st & (dv > T)).sum()),
            "onset_hazard": hazard, "onset_hazard_entries": n_fresh}


def teacher_forced(path):
    if not os.path.exists(path):
        return None
    Q = json.load(open(path))["queries_detail"]
    return [{"unit": q["unit"], "stop": bool(q["stop"]), "dev": float(q["dev"])} for q in Q if q["dist"] > FAR_M]


def closed_loop(d, ref, collect, val):
    out = {"all": [], "on_path": [], "off_path": [], "mid_path": []}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        unit = Path(f).stem.replace("trace_", "").replace("_r", "#")
        pose, act, dist = z["pose"], z["act"], z["dist"]
        path = None if val else expert_nav(collect, unit)
        if not val and path is None:
            continue
        prev = (False, 0.0)
        for t in range(0, len(act) - H_EXEC + 1, H_EXEC):
            c = act[t:t + H_EXEC]
            dev = max(float(np.linalg.norm(c[:, LXYZ] - ref["left"], axis=1).max()), float(np.linalg.norm(c[:, RXYZ] - ref["right"], axis=1).max()))
            q = {"unit": unit, "t": t, "stop": bool(np.abs(c[:, 16]).mean() < STOP_V and np.abs(c[:, 17]).mean() < STOP_W), "dev": dev,
                 "prev_stop": prev[0], "prev_dev": prev[1]}
            prev = (q["stop"], dev)
            if t < TUCK_SKIP:
                continue
            if val:
                if dist[t] > FAR_M_VAL:
                    out["all"].append(q)
                continue
            if np.linalg.norm(pose[t, :2] - path[-1]) <= FAR_M:          # same distance as the expert side
                continue
            pdev = float(np.min(np.linalg.norm(path - pose[t, :2], axis=1)))
            q["path_dev"] = pdev
            out["all"].append(q)
            out["on_path" if pdev < 0.5 else "off_path" if pdev >= 1.0 else "mid_path"].append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help="NAME=TRIGGER_DIR:TRACES_TRAIN:TRACES_VAL")
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--collect", default="/workspace/runtime/runs/collect_clean_v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    rep = {"definition": __doc__, "thresholds_m": THRESHOLDS, "far_m": FAR_M, "far_m_val": FAR_M_VAL, "tuck_skip": TUCK_SKIP, "models": {}}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        tdir, tr_train, tr_val = rest.split(":")
        conds = {"expert_obs_heldout_val": teacher_forced(f"{tdir}/heldout_val.json"),
                 "expert_obs_train_units": teacher_forced(f"{tdir}/train_units.json")}
        cl = closed_loop(tr_train, ref, a.collect, False) if os.path.isdir(tr_train) else None
        for k in ("all", "on_path", "mid_path", "off_path"):
            conds[f"closed_loop_train_units_{k}"] = cl[k] if cl else None
        clv = closed_loop(tr_val, ref, a.collect, True) if os.path.isdir(tr_val) else None
        conds["closed_loop_ovmm_val"] = clv["all"] if clv else None
        rep["models"][name] = {c: ({f"{T:.2f}": cats(q, T) for T in THRESHOLDS} if q is not None else "missing") for c, q in conds.items()}
        if cl:                                                       # first far early reach per unit (15 cm)
            first = {}
            for q in cl["all"]:
                if q["stop"] and q["dev"] > 0.15 and q["unit"] not in first:
                    first[q["unit"]] = q
            rep["models"][name]["first_far_early_reach_train_units"] = {
                "units": len(first), "of": len({q["unit"] for q in cl["all"]}),
                "on_path_lt_0.5m": sum(q["path_dev"] < 0.5 for q in first.values()),
                "mid_0.5_1m": sum(0.5 <= q["path_dev"] < 1.0 for q in first.values()),
                "off_path_ge_1m": sum(q["path_dev"] >= 1.0 for q in first.values()),
                "step_med": int(np.median([q["t"] for q in first.values()])) if first else None}
        for c, x in rep["models"][name].items():
            if c.startswith("first_"):
                print(f"{name:10s} {c:36s} {json.dumps(x)}", flush=True); continue
            if x == "missing":
                print(f"{name:10s} {c:36s} missing", flush=True); continue
            print(f"{name:10s} {c:36s} n={x['0.15']['queries']:5d} units={x['0.15'].get('units', 0):3d} | onset_hazard "
                  + " ".join(f"{T}:{x[T].get('onset_hazard')}" for T in x) + " | per-query early_reach "
                  + " ".join(f"{T}:{x[T]['early_reach']:.3f}" for T in x) + f" | reach_moving@.15 {x['0.15'].get('reach_moving', 0):.3f}"
                  f" pure_stop {x['0.15'].get('pure_stop', 0):.3f} stop_tuck {x['0.15'].get('stop_tuck_like', 0):.3f}"
                  f" interm@.15 {x['0.15'].get('intermediate', 0):.3f} units_w_reach@.15 {x['0.15'].get('units_with_early_reach', 0)}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
