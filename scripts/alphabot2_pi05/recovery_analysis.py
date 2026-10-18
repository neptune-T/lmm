#!/usr/bin/env python3
"""Analysis of recovery_eval.py (CPU, read-only). Per root, expert vs the matched v2-B continuations.

Common outcome (same for expert and policy): APPROACH = the target object within 1.0 m within the budget (expert:
its navigation; policy: POLICY_STEPS); time to approach; distance to the target at the end of navigation.
Expert, additionally: navigation arrival at its re-planned standoff (nav.follow "arrived"), stage reached, task
funnel at the end (find / grasp / place / full), collision steps and navmesh filtering during navigation, IK-invalid
labels, arms tucked (max arm offset from the expert NAV pose over the navigation labels), corrective NAV
demonstrations = labelled tuck + navigate_to_object steps from the off-path root, "clean" when the navigation
arrived with no collision step, arms tucked (<= 8 cm) and every label IK-valid.
Policy, additionally: false reach while > 2 m from the expert stop point, stall steps, contact steps, funnel find.
Paired: expert-only / policy-only / both / neither approach per root (exact sign test), by category and group.
    recovery_analysis.py --run DIR --roots roots.json --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, math, os
from pathlib import Path

import numpy as np

H_EXEC, STOP_V, STOP_W, MOVE_V, MOVE_W, DT, REACH_M, TUCK_MAX, FAR_M, APPROACH_M = 5, 0.03, 0.05, 0.03, 0.05, 0.1, 0.15, 0.08, 2.0, 1.0
NAV_STAGES = ("tuck", "navigate_to_object")
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def sign_p(w, l):
    n = w + l
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n), 4) if n else None


def arm_dev(act, ref):
    return np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))


def expert_metrics(rec, z, ref):
    stage = [str(s) for s in z["stage"]]; dist = z["dist"]; act = z["act"]
    nav_idx = [i for i, s in enumerate(stage) if s in NAV_STAGES]
    end_nav = (max(nav_idx) + 1) if nav_idx else 0
    log = rec.get("expert_log") or {}; nv = log.get("nav_to_object") or {}
    appr = [i for i in range(end_nav) if dist[i] <= APPROACH_M]
    dev = arm_dev(act[:end_nav], ref) if end_nav else np.zeros(0)
    ik_ok = np.asarray(z["ik_ok"][:end_nav], bool) if end_nav else np.zeros(0, bool)
    arrived = bool(nv.get("arrived"))
    clean = bool(arrived and not nv.get("collision_steps") and (dev.max() if len(dev) else 0) <= TUCK_MAX and ik_ok.all())
    reached = [s for _, s in rec.get("expert_events", [])]
    return {"approach": bool(appr), "steps_to_approach": appr[0] if appr else None, "nav_steps": end_nav,
            "nav_arrived": arrived, "arrival_dist_to_target_m": round(float(dist[end_nav - 1]), 3) if end_nav else None,
            "collision_steps": nv.get("collision_steps"), "arm_dev_max_m": round(float(dev.max()), 4) if len(dev) else None,
            "ik_invalid_nav": int((~ik_ok).sum()), "status": rec.get("expert_status"), "stages_reached": reached,
            "funnel_end": rec.get("funnel_end"), "demo_steps": end_nav, "clean": clean}


def policy_metrics(rec, z, ref, stop):
    pose, act, dist = z["pose"], z["act"], z["dist"]
    dstop = np.linalg.norm(pose[:, :2] - stop, axis=1)
    d = np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]); yaw = pose[:, 2]
    moving = (np.linalg.norm(d, axis=1) / DT >= MOVE_V) | (np.abs(np.angle(np.exp(1j * np.diff(yaw, prepend=yaw[:1])))) / DT >= MOVE_W)
    dev = arm_dev(act, ref)
    fr = None
    for t in range(0, len(act) - H_EXEC + 1, H_EXEC):
        if dstop[t] > FAR_M and np.abs(act[t:t + H_EXEC, 16]).mean() < STOP_V and np.abs(act[t:t + H_EXEC, 17]).mean() < STOP_W \
                and dev[t:t + H_EXEC].max() > REACH_M:
            fr = t; break
    appr = np.flatnonzero(dist <= APPROACH_M)
    return {"approach": bool(len(appr)), "steps_to_approach": int(appr[0]) if len(appr) else None,
            "min_dist_to_target_m": round(float(dist.min()), 3), "false_reach": fr is not None, "false_reach_step": fr,
            "stall_steps": int((~moving).sum()), "contact_steps": int((np.asarray(z["contacts"]) > 0).sum()),
            "funnel_end": rec.get("funnel_end"), "reproduction_err_m": rec.get("reproduction_max_pos_err_m_first50")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    roots = {r["root_id"]: r for r in json.load(open(a.roots))["roots"]}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.run}/recovery_w*.jsonl")) for l in open(f)]
    per = {}
    for rec in rows:
        rid, mode = rec["root_id"], rec["mode"]
        e = per.setdefault(rid, {"root": {k: roots[rid][k] for k in ("category", "group", "unit", "base_episode", "r", "path_dev_m",
                                                                      "dist_to_stop_m", "progress_last50_m")}})
        if "trace_out" not in rec:
            e[mode] = {"skipped": rec.get("skipped"), "engineering_error": rec.get("engineering_error"), "root_check": rec.get("root_check")}
            continue
        z = np.load(rec["trace_out"], allow_pickle=True)
        e["root_check"] = rec.get("root_check")
        if mode == "expert":
            e[mode] = expert_metrics(rec, z, ref)
        else:
            md = np.load(Path(roots[rid]["source"]) / "metadata.npz")
            nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
            e[mode] = policy_metrics(rec, z, ref, md["base_xz_yaw"][nav[-1], :2])
    ok = {rid: e for rid, e in per.items() if all(m in e and "approach" in e[m] for m in ("expert", "policy_orig", "policy_alt"))}
    rep = {"roots_total": len(per), "roots_complete": len(ok),
           "incomplete": {rid: {m: e.get(m) for m in ("expert", "policy_orig", "policy_alt") if not (m in e and "approach" in e[m])}
                          for rid, e in per.items() if rid not in ok},
           "root_check_max_pos_m": max((e["root_check"]["pos_m"] for e in per.values() if e.get("root_check")), default=None),
           "reproduction_err_m": sorted(e["policy_orig"]["reproduction_err_m"] for e in ok.values() if e["policy_orig"]["reproduction_err_m"] is not None),
           "by_split": {}}
    splits = {"all": list(ok.values())}
    for c in ("pre_false_reach", "near_stall", "off_path_no_fr"):
        splits[c] = [e for e in ok.values() if e["root"]["category"] == c]
    for g in ("train_unit", "heldout"):
        splits[g] = [e for e in ok.values() if e["root"]["group"] == g]
    for name, E in splits.items():
        if not E:
            continue
        X = [e["expert"] for e in E]
        s = {"n": len(E), "base_episodes": len({e["root"]["base_episode"] for e in E}),
             "expert": {"approach": sum(x["approach"] for x in X), "nav_arrived": sum(x["nav_arrived"] for x in X),
                        "clean_demo_roots": sum(x["clean"] for x in X),
                        "completed_task": sum(bool(x["funnel_end"] and x["funnel_end"].get("full")) for x in X),
                        "grasped": sum(bool(x["funnel_end"] and x["funnel_end"].get("grasp")) for x in X),
                        "failures": [x["status"] for x in X if x["status"] != "completed"],
                        "steps_to_approach_med": int(np.median([x["steps_to_approach"] for x in X if x["steps_to_approach"] is not None]))
                        if any(x["steps_to_approach"] is not None for x in X) else None,
                        "collision_roots": sum(bool(x["collision_steps"]) for x in X),
                        "arm_dev_max_m": max((x["arm_dev_max_m"] or 0) for x in X),
                        "ik_invalid_nav_total": sum(x["ik_invalid_nav"] for x in X),
                        "demo_steps_total": sum(x["demo_steps"] for x in X),
                        "clean_demo_steps_total": sum(x["demo_steps"] for x in X if x["clean"])}}
        for m in ("policy_orig", "policy_alt"):
            P = [e[m] for e in E]
            s[m] = {"approach": sum(p["approach"] for p in P), "false_reach": sum(p["false_reach"] for p in P),
                    "find": sum(bool(p["funnel_end"] and p["funnel_end"].get("find")) for p in P),
                    "contact_roots": sum(p["contact_steps"] > 0 for p in P),
                    "steps_to_approach_med": int(np.median([p["steps_to_approach"] for p in P if p["steps_to_approach"] is not None]))
                    if any(p["steps_to_approach"] is not None for p in P) else None}
            w = sum(e["expert"]["approach"] and not e[m]["approach"] for e in E); l = sum(e[m]["approach"] and not e["expert"]["approach"] for e in E)
            s[f"paired_vs_{m}"] = {"expert_only": w, "policy_only": l, "both": sum(e["expert"]["approach"] and e[m]["approach"] for e in E),
                                   "neither": sum(not e["expert"]["approach"] and not e[m]["approach"] for e in E), "sign_test_p": sign_p(w, l)}
        rep["by_split"][name] = s
    rep["per_root"] = per
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("roots_total", "roots_complete", "root_check_max_pos_m")}), "reproduction_err_m", rep["reproduction_err_m"][-5:])
    for name, s in rep["by_split"].items():
        print(f"== {name}: {json.dumps(s)}")
    if rep["incomplete"]:
        print("incomplete:", json.dumps(rep["incomplete"])[:1500])


if __name__ == "__main__":
    main()
