#!/usr/bin/env python3
"""Low-cost statistic (CPU, read-only): how often does the EXPERT rotate in place while still far from the target,
and are those states more likely to trigger a false reach?

Expert frames: navigate_to_object frames > FAR_M from the expert's stop point in the v2 TRAIN sources (manifest
order), classified by the expert's own executed base command: rotate_in_place (|v| < 0.03 and |omega| >= 0.05),
drive (|v| >= 0.03), still (neither).
Link to false reach:
  teacher forced  the saved expert_trigger queries (A, B, sanity_v1; held-out and training units): false-reach rate
                  (15 cm) on queries whose expert frame is rotate_in_place vs drive
  closed loop     v2-B no-perturbation rollouts of the perturbation run: start states that are rotate_in_place vs
                  drive; false reach anywhere in the rollout and early (< 100 steps); exact 2x2 counts + Fisher p
    expert_rotation.py --manifest v2.json --collect DIR --trigger RUNS/v2/trigger --perturb RUNS/v2/perturb/full \
        --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, math, os
from pathlib import Path

import numpy as np

FAR_M, STOP_V, STOP_W, REACH_M, H_EXEC, EARLY = 2.0, 0.03, 0.05, 0.15, 5, 100
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def cls(v, w):
    return "drive" if abs(v) >= STOP_V else ("rotate_in_place" if abs(w) >= STOP_W else "still")


def expert_frames(src):
    md = np.load(Path(src) / "metadata.npz"); act = np.load(Path(src) / "lowdim.npz")["action"]
    st = [str(x) for x in md["stage"]]
    nav = [t for t, s in enumerate(st) if s == "navigate_to_object"]
    if not nav:
        return {}
    xz = md["base_xz_yaw"][:, :2]; stop = xz[nav[-1]]
    return {t: (cls(act[t, 16], act[t, 17]), float(np.linalg.norm(xz[t] - stop))) for t in nav}


def fisher_p(a, b, c, d):
    """two-sided Fisher exact p for [[a, b], [c, d]]"""
    n1, n2, k, n = a + b, c + d, a + c, a + b + c + d
    pr = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    p0 = pr(a)
    return round(min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= p0 * (1 + 1e-9))), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--collect", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--perturb", required=True)
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    units = list(json.load(open(a.manifest))["train"])
    if a.max_episodes:
        units = units[: a.max_episodes]
    cnt = {"rotate_in_place": 0, "drive": 0, "still": 0}; ep_rot = 0; seg_far = []
    for u in units:
        ep, rep = u.split("#")
        src = f"{a.collect}/episodes/{ep}_r{rep}"
        if not os.path.exists(f"{src}/metadata.npz"):
            continue
        F = {t: c for t, (c, d) in expert_frames(src).items() if d > FAR_M}
        for c in F.values():
            cnt[c] += 1
        ep_rot += any(c == "rotate_in_place" for c in F.values())
        run = 0                                                         # in-place rotation segments while far
        for t in sorted(F):
            run = run + 1 if F[t] == "rotate_in_place" else (seg_far.append(run) or 0 if run else 0)
        if run:
            seg_far.append(run)
    tot = sum(cnt.values())
    rep = {"expert_far_nav_frames": tot, "share": {k: round(v / max(1, tot), 4) for k, v in cnt.items()}, "counts": cnt,
           "episodes": len(units), "episodes_with_far_in_place_rotation": ep_rot,
           "far_rotation_segments": len(seg_far), "far_rotation_segment_steps_median": int(np.median(seg_far)) if seg_far else None}
    # teacher forced
    rep["teacher_forced"] = {}
    for f in sorted(glob.glob(f"{a.trigger}/*/*.json")):
        name = f"{Path(f).parent.name}:{Path(f).stem}"
        Q = [q for q in json.load(open(f))["queries_detail"] if q["dist"] > FAR_M]
        cache, res = {}, {}
        for q in Q:
            u = q["unit"]
            if u not in cache:
                ep, rp = u.split("#"); cache[u] = expert_frames(f"{a.collect}/episodes/{ep}_r{rp}")
            c = cache[u].get(q["t"], (None,))[0]
            trig = bool(q["stop"]) and float(q["dev"]) > REACH_M
            r = res.setdefault(c, [0, 0]); r[0] += trig; r[1] += 1
        rot, drv = res.get("rotate_in_place", [0, 0]), res.get("drive", [0, 0])
        rep["teacher_forced"][name] = {k: {"false_reach": v[0], "queries": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None} for k, v in res.items() if k}
        rep["teacher_forced"][name]["fisher_p_rotate_vs_drive"] = fisher_p(rot[0], rot[1] - rot[0], drv[0], drv[1] - drv[0])
    # closed loop: v2-B no-perturbation rollouts by start-state class
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.perturb}/perturb_w*.jsonl")) for l in open(f)]
    cl = {}
    for r in rows:
        if r["level"] != "none" or "steps" not in r:
            continue
        fr = expert_frames(r["source"]); c = fr.get(r["t"], (None,))[0]
        z = np.load(f"{a.perturb}/traces/{r['job']}.npz"); act, pose = z["act"], z["pose"]
        md = np.load(Path(r["source"]) / "metadata.npz")
        nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
        dstop = np.linalg.norm(pose[:, :2] - md["base_xz_yaw"][nav[-1], :2], axis=1)
        dev = np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))
        trig = next((t for t in range(0, len(act) - H_EXEC + 1, H_EXEC) if dstop[t] > FAR_M and np.abs(act[t:t + H_EXEC, 16]).mean() < STOP_V
                     and np.abs(act[t:t + H_EXEC, 17]).mean() < STOP_W and dev[t:t + H_EXEC].max() > REACH_M), None)
        e = cl.setdefault(c, {"rollouts": 0, "false_reach": 0, "early_false_reach": 0, "units": set()})
        e["rollouts"] += 1; e["false_reach"] += trig is not None; e["early_false_reach"] += trig is not None and trig < EARLY
        e["units"].add(f"{r['unit']}@{r['t']}")
    rot, drv = cl.get("rotate_in_place", {}), cl.get("drive", {})
    rep["closed_loop_B_none"] = {k: {**{kk: v for kk, v in e.items() if kk != "units"}, "states": len(e["units"])} for k, e in cl.items() if k}
    if rot and drv:
        rep["closed_loop_B_none"]["fisher_p_false_reach"] = fisher_p(rot["false_reach"], rot["rollouts"] - rot["false_reach"],
                                                                      drv["false_reach"], drv["rollouts"] - drv["false_reach"])
        rep["closed_loop_B_none"]["fisher_p_early_false_reach"] = fisher_p(rot["early_false_reach"], rot["rollouts"] - rot["early_false_reach"],
                                                                            drv["early_false_reach"], drv["rollouts"] - drv["early_false_reach"])
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
