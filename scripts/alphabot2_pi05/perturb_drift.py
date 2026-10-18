#!/usr/bin/env python3
"""EXPLORATORY (not part of the pre-registered perturbation decision, which stays INCONCLUSIVE): does spontaneous
closed-loop heading / path drift precede premature reaching? CPU only, existing traces of the "none" level
(rollouts started from exact expert NAV states, no imposed perturbation).

Thresholds fixed before looking at these results:
  heading error   |robot yaw - expert yaw at the nearest expert navigate_to_object pose|   > 30 deg (20 / 45)
  path deviation  distance to the expert navigate_to_object path                          > 0.5 m (0.25 / 1.0)
  drift onset     first step from which the condition holds for SUSTAIN = 10 consecutive steps
  false reach     first far query (> 2 m from the expert stop point) with base stop and arm > 15 cm off the NAV pose
  base-only stall >= 30 consecutive far steps not moving, arm within 8 cm, commanded |v|, |omega| ~ 0
  progressing     distance to the stop point fell >= 0.25 m over the 50 steps before the event
Per triggering rollout: drift onsets and their lead (steps before the trigger). Comparisons:
  time-matched    share of triggering rollouts already drifted at their trigger time t* vs the share of
                  non-triggering rollouts already drifted at the same t* (averaged over the triggers)
  exposure hazard false-reach events per eligible far query (every 5 steps, up to the first trigger), by the
                  CURRENT drift state at that query (drifted vs on track)
Cluster bootstrap 95 % CIs (2000 resamples) resample BASE OVMM episodes (all replicas / states / seeds of an episode together).
    perturb_drift.py --run RUNS/v2/perturb/full --subset yaw --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, math, os
from pathlib import Path

import numpy as np

H_EXEC, STOP_V, STOP_W, MOVE_V, MOVE_W, DT = 5, 0.03, 0.05, 0.03, 0.05, 0.1
FAR_M, REACH_M, TUCK_MAX, SUSTAIN, STALL_RUN, PROG_WIN, PROG_M = 2.0, 0.15, 0.08, 10, 30, 50, 0.25
HEAD_DEG = (20.0, 30.0, 45.0); PATH_M = (0.25, 0.5, 1.0); PRIMARY = (30.0, 0.5)
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def onset_sustained(cond):
    run = 0
    for t, c in enumerate(cond):
        run = run + 1 if c else 0
        if run >= SUSTAIN:
            return t - SUSTAIN + 1
    return None


def analyse(trace, src, ref):
    z = np.load(trace); pose, act = z["pose"], z["act"]
    md = np.load(Path(src) / "metadata.npz")
    nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
    exz, eyaw = md["base_xz_yaw"][nav, :2], md["base_xz_yaw"][nav, 2]
    stop = exz[-1]
    d_all = np.linalg.norm(pose[:, None, :2] - exz[None], axis=2)
    near = d_all.argmin(1)
    path_dev = d_all.min(1)
    head_err = np.degrees(np.abs(np.angle(np.exp(1j * (pose[:, 2] - eyaw[near])))))
    dstop = np.linalg.norm(pose[:, :2] - stop, axis=1)
    dd = np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]); yaw = pose[:, 2]
    moving = (np.linalg.norm(dd, axis=1) / DT >= MOVE_V) | (np.abs(np.angle(np.exp(1j * np.diff(yaw, prepend=yaw[:1])))) / DT >= MOVE_W)
    dev = np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))
    cmd0 = (np.abs(act[:, 16]) < STOP_V) & (np.abs(act[:, 17]) < STOP_W)
    far = dstop > FAR_M
    trig, queries = None, []
    for t in range(0, len(act) - H_EXEC + 1, H_EXEC):
        if not far[t]:
            continue
        c = act[t:t + H_EXEC]
        hit = bool(np.abs(c[:, 16]).mean() < STOP_V and np.abs(c[:, 17]).mean() < STOP_W and dev[t:t + H_EXEC].max() > REACH_M)
        queries.append((t, hit))
        if hit:
            trig = t; break
    bstall = far & ~moving & (dev <= TUCK_MAX) & cmd0
    b_on, run = None, 0
    for t, c in enumerate(bstall):
        run = run + 1 if c else 0
        if run >= STALL_RUN:
            b_on = t - STALL_RUN + 1; break
    stall = far & ~moving
    out = {"steps": len(act), "trigger": trig, "base_only_stall_onset": b_on, "queries": queries,
           "stall_steps": int(stall.sum()), "stall_false_reach": int((stall & (dev > REACH_M)).sum()),
           "stall_base_only": int((stall & (dev <= TUCK_MAX) & cmd0).sum()),
           "stall_blocked": int((stall & (dev <= TUCK_MAX) & ~cmd0).sum()),
           "head_err": head_err, "path_dev": path_dev, "dstop": dstop, "moving": moving}
    for h in HEAD_DEG:
        out[f"head{h:g}"] = onset_sustained(head_err > h)
    for p in PATH_M:
        out[f"path{p:g}"] = onset_sustained(path_dev > p)
    for ev, t in (("trigger", trig), ("bstall", b_on)):
        if t is not None:
            w0 = max(0, t - PROG_WIN)
            out[f"{ev}_progressing"] = bool(dstop[w0] - dstop[t] >= PROG_M)
            out[f"{ev}_moving_frac_before"] = float(moving[w0:t].mean()) if t > w0 else None
            out[f"{ev}_head_err_at"] = float(head_err[t]); out[f"{ev}_path_dev_at"] = float(path_dev[t])
    return out


def lead(o, key, t):
    on = o[key]
    return None if on is None or on > t else t - on


def stats(R, h, p):
    """primary quantities for heading threshold h (deg) and path threshold p (m)"""
    hk, pk = f"head{h:g}", f"path{p:g}"
    T = [o for o in R if o["trigger"] is not None]; N = [o for o in R if o["trigger"] is None]
    res = {"n": len(R), "triggered": len(T)}
    for key in (hk, pk, "either"):
        def drifted_by(o, t):
            if key == "either":
                return any(o[k] is not None and o[k] <= t for k in (hk, pk))
            return o[key] is not None and o[key] <= t
        p_trig = float(np.mean([drifted_by(o, o["trigger"]) for o in T])) if T else None
        p_ctrl = float(np.mean([np.mean([drifted_by(n, o["trigger"]) for n in N]) for o in T])) if T and N else None
        leads = [(o["trigger"] - min(o[k] for k in (hk, pk) if o[k] is not None and o[k] <= o["trigger"])) if key == "either"
                 else lead(o, key, o["trigger"]) for o in T if drifted_by(o, o["trigger"])]
        res[key] = {"triggered_drifted_before": int(sum(drifted_by(o, o["trigger"]) for o in T)),
                    "p_triggered": round(p_trig, 3) if p_trig is not None else None,
                    "p_time_matched_nontriggered": round(p_ctrl, 3) if p_ctrl is not None else None,
                    "lead_steps_median": int(np.median(leads)) if leads else None,
                    "lead_steps_iqr": [int(np.percentile(leads, 25)), int(np.percentile(leads, 75))] if leads else None}
    # exposure hazard by current drift state (either threshold, at the query step)
    ev = {"drifted": [0, 0], "on_track": [0, 0]}
    for o in R:
        for t, hit in o["queries"]:
            s = "drifted" if (o["head_err"][t] > h or o["path_dev"][t] > p) else "on_track"
            ev[s][1] += 1; ev[s][0] += hit
    res["hazard"] = {s: {"events": e, "eligible": x, "rate": round(e / x, 5) if x else None} for s, (e, x) in ev.items()}
    rd, ro = res["hazard"]["drifted"]["rate"], res["hazard"]["on_track"]["rate"]
    res["hazard_ratio_drifted_vs_on_track"] = round(rd / ro, 3) if rd is not None and ro else None
    res["triggered_progressing"] = int(sum(bool(o.get("trigger_progressing")) for o in T))
    return res


def cluster_ci(R, groups, h, p, rng, n_boot=2000):
    keys = sorted(set(groups))
    idx = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    diffs, hrs, ptr = [], [], []
    for _ in range(n_boot):
        pick = [i for k in rng.choice(keys, len(keys), replace=True) for i in idx[k]]
        S = stats([R[i] for i in pick], h, p)
        e = S["either"]
        if e["p_triggered"] is not None and e["p_time_matched_nontriggered"] is not None:
            diffs.append(e["p_triggered"] - e["p_time_matched_nontriggered"]); ptr.append(e["p_triggered"])
        if S["hazard_ratio_drifted_vs_on_track"] is not None:
            hrs.append(S["hazard_ratio_drifted_vs_on_track"])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None
    return {"either_p_triggered_minus_time_matched_ci95": q(diffs), "either_p_triggered_ci95": q(ptr),
            "hazard_ratio_ci95": q(hrs), "n_boot": n_boot, "clusters": len(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--subset", default="yaw", help="'yaw' = the yaw common subset (state, seed with all yaw levels valid)")
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.run}/perturb_w*.jsonl")) for l in open(f)]
    ok = {(r["unit"], r["t"], r["seed"], r["level"]): r for r in rows if "steps" in r and os.path.exists(f"{a.run}/traces/{r['job']}.npz")}
    lv = ["none", "yaw15", "yaw30", "yaw60"] if a.subset == "yaw" else ["none", "pos0.25", "pos0.5", "pos1.0"]
    keys = sorted({k[:3] for k in ok if all((*k[:3], L) in ok for L in lv)})
    R, groups, meta = [], [], []
    for k in keys:
        r = ok[(*k, "none")]
        R.append(analyse(f"{a.run}/traces/{r['job']}.npz", r["source"], ref))
        groups.append(k[0].split("#")[0]); meta.append({"unit": k[0], "t": k[1], "seed": k[2], "group": r["group"]})
    rng = np.random.default_rng(0)
    rep = {"label": "EXPLORATORY follow-up; the pre-registered perturbation verdict stays INCONCLUSIVE", "definition": __doc__,
           "subset": a.subset, "rollouts": len(R), "base_episodes": len(set(groups)), "primary": {}, "sensitivity": {}}
    h0, p0 = PRIMARY
    rep["primary"] = {**stats(R, h0, p0), "cluster_bootstrap": cluster_ci(R, groups, h0, p0, rng)}
    for h in HEAD_DEG:
        for p in PATH_M:
            if (h, p) != PRIMARY:
                s = stats(R, h, p)
                rep["sensitivity"][f"head{h:g}_path{p:g}"] = {"either": s["either"], "hazard_ratio": s["hazard_ratio_drifted_vs_on_track"], "hazard": s["hazard"]}
    # stalls: false-reach vs base-only vs blocked, and base-only stall onsets
    tot = {k: sum(o[k] for o in R) for k in ("stall_steps", "stall_false_reach", "stall_base_only", "stall_blocked")}
    B = [o for o in R if o["base_only_stall_onset"] is not None]
    rep["stalls"] = {"totals": tot, "shares": {k: round(tot[k] / max(1, tot["stall_steps"]), 3) for k in ("stall_false_reach", "stall_base_only", "stall_blocked")},
                     "rollouts_with_base_only_stall_run": len(B),
                     "rollouts_with_false_reach": sum(o["trigger"] is not None for o in R),
                     "rollouts_with_both": sum(o["trigger"] is not None and o["base_only_stall_onset"] is not None for o in R),
                     "base_only_stall_before_false_reach": sum(o["trigger"] is not None and o["base_only_stall_onset"] is not None
                                                               and o["base_only_stall_onset"] < o["trigger"] for o in R),
                     "base_only_stall_drifted_before_30deg_or_0.5m": sum(any(o[k] is not None and o[k] <= o["base_only_stall_onset"] for k in ("head30", "path0.5")) for o in B),
                     "base_only_stall_progressing": sum(bool(o.get("bstall_progressing")) for o in B)}
    rep["per_episode"] = [{**m, "trigger": o["trigger"], "head30_onset": o["head30"], "path0.5_onset": o["path0.5"],
                           "trigger_progressing": o.get("trigger_progressing"), "trigger_head_err_deg": o.get("trigger_head_err_at"),
                           "trigger_path_dev_m": o.get("trigger_path_dev_at"), "base_only_stall_onset": o["base_only_stall_onset"],
                           "stall_steps": o["stall_steps"], "stall_false_reach": o["stall_false_reach"], "stall_base_only": o["stall_base_only"]}
                          for m, o in zip(meta, R)]
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    P = rep["primary"]
    print(json.dumps({"rollouts": len(R), "base_episodes": rep["base_episodes"], "triggered": P["triggered"],
                      "heading30": P["head30"], "path0.5": P["path0.5"], "either": P["either"], "hazard": P["hazard"],
                      "hazard_ratio": P["hazard_ratio_drifted_vs_on_track"], "triggered_progressing": P["triggered_progressing"],
                      "cluster_bootstrap": P["cluster_bootstrap"]}, indent=1))
    for k, s in rep["sensitivity"].items():
        e = s["either"]
        print(f"sens {k:16s} drifted_before {e['triggered_drifted_before']} p_trig {e['p_triggered']} p_ctrl {e['p_time_matched_nontriggered']} lead_med {e['lead_steps_median']} HR {s['hazard_ratio']}")
    print("stalls", json.dumps(rep["stalls"]))


if __name__ == "__main__":
    main()
