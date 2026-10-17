#!/usr/bin/env python3
"""Closed-loop counterpart of expert_trigger.py: the same false "base stop + arm move" trigger, but on the
policy's OWN observations (eval traces), split by how far the robot has drifted from the expert's path for the
same unit (eval units start from the expert's seeded reset, so step 0 is an expert state).

Per query (every H_EXEC steps of the trace): stop / arm_move / trigger exactly as expert_trigger.classify, on the
executed chunk rows; distance = trace distance to the target object; path deviation = distance from the robot to
the nearest base position of the expert's navigate_to_object segment of that unit (collect_clean_v1 source).
    cl_trigger.py --traces A_train=DIR ... --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_trigger import H_EXEC, classify  # noqa: E402

DIST_BINS = [(0, 2), (2, 4), (4, 8), (8, 99)]
DEV_BINS = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]


def expert_path(collect, unit):
    ep, rep = unit.split("#")
    d = Path(collect) / "episodes" / f"{ep}_r{rep}"
    if not (d / "metadata.npz").exists():
        return None
    md = np.load(d / "metadata.npz")
    st = [str(x) for x in md["stage"]]
    nav = [t for t, s in enumerate(st) if s == "navigate_to_object"]
    return md["base_xz_yaw"][nav, :2] if nav else None


def rate(qs, key="trig"):
    return round(float(np.mean([q[key] for q in qs])), 4) if qs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="LABEL=DIR")
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--collect", default="/workspace/runtime/runs/collect_clean_v1")
    ap.add_argument("--far-m", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    rep = {"far_m": a.far_m, "runs": {}}
    for spec in a.traces:
        lab, d = spec.split("=", 1)
        Q, starts = [], []
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            unit = Path(f).stem.replace("trace_", "").replace("_r", "#")
            path = expert_path(a.collect, unit)
            pose, act, dist = z["pose"], z["act"], z["dist"]
            if path is not None:
                starts.append(float(np.linalg.norm(pose[0, :2] - path[0])))
            first_trig = None
            for t in range(0, len(act) - H_EXEC + 1, H_EXEC):
                stop, arm, dev_arm = classify(act[t:t + H_EXEC], ref)
                pdev = float(np.min(np.linalg.norm(path - pose[t, :2], axis=1))) if path is not None else None
                q = {"unit": unit, "t": t, "dist": float(dist[t]), "path_dev": pdev, "stop": stop, "arm": arm, "trig": stop and arm}
                if q["trig"] and q["dist"] > a.far_m and first_trig is None:
                    first_trig = q
                Q.append(q)
        far = [q for q in Q if q["dist"] > a.far_m]
        r = {"dir": d, "units": len({q["unit"] for q in Q}), "queries": len(Q),
             "expert_path_found": sum(1 for q in {q["unit"]: q for q in Q}.values() if q["path_dev"] is not None),
             "start_offset_from_expert_m_max": round(max(starts), 4) if starts else None,
             "far": {"queries": len(far), "trigger": rate(far), "stop": rate(far, "stop"), "arm_move": rate(far, "arm")},
             "far_by_dist": {}, "far_by_path_dev": {}}
        for lo, hi in DIST_BINS:
            s = [q for q in Q if lo <= q["dist"] < hi]
            if s:
                r["far_by_dist"][f"{lo}-{hi}m"] = {"queries": len(s), "trigger": rate(s), "stop": rate(s, "stop"), "arm_move": rate(s, "arm")}
        for lo, hi in DEV_BINS:
            s = [q for q in far if q["path_dev"] is not None and lo <= q["path_dev"] < hi]
            if s:
                r["far_by_path_dev"][f"{lo}-{hi}m"] = {"queries": len(s), "trigger": rate(s), "stop": rate(s, "stop"), "arm_move": rate(s, "arm"),
                                                       "units": len({q["unit"] for q in s})}
        # where the FIRST far trigger of each unit happens: on the expert path or already off it
        firsts = {}
        for q in Q:
            if q["trig"] and q["dist"] > a.far_m and q["unit"] not in firsts:
                firsts[q["unit"]] = q
        r["first_far_trigger"] = {"units": len(firsts),
                                  "on_path_lt_0.5m": sum(1 for q in firsts.values() if q["path_dev"] is not None and q["path_dev"] < 0.5),
                                  "off_path_ge_1m": sum(1 for q in firsts.values() if q["path_dev"] is not None and q["path_dev"] >= 1.0),
                                  "step_med": int(np.median([q["t"] for q in firsts.values()])) if firsts else None,
                                  "path_dev_med_m": round(float(np.median([q["path_dev"] for q in firsts.values() if q["path_dev"] is not None])), 2)
                                  if any(q["path_dev"] is not None for q in firsts.values()) else None}
        rep["runs"][lab] = r
        print(f"== {lab}: {json.dumps(r)}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
