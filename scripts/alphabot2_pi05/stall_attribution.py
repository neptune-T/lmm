#!/usr/bin/env python3
"""How much actual stall time do false reaches explain? (CPU, read-only: closed-loop eval traces.)

Stall step: far-NAV step (target object > FAR_M away, after TUCK_SKIP steps) on which the base does not actually
move (|v| < 0.03 m/s and |omega| < 0.05 rad/s from the pose). Each stall step is attributed by what was commanded
on that step:
  false_reach   arm commanded > T from the expert NAV pose (reach-like; expert tuck <= 0.06 m, reach >= 0.25 m)
  base_only     commanded |v| < 0.03 and |omega| < 0.05 and arm within TUCK_MAX of the NAV pose
  blocked       commanded |v| >= 0.03 or |omega| >= 0.05 with arm within TUCK_MAX (executor / collision)
  intermediate  arm between TUCK_MAX and T
Shares of total stall time (pooled) with an episode-level bootstrap 95 % CI, per threshold T in 10/15/20 cm;
plus the share of episodes whose stall time is mostly false reach.
    stall_attribution.py --traces B_train=DIR B_val=DIR ... --expert-pose stall_diag.json --out OUT.json
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

DT, MOVE_V, MOVE_W, CMD_V, CMD_W = 0.1, 0.03, 0.05, 0.03, 0.05
THRESHOLDS, TUCK_MAX, FAR_M, TUCK_SKIP = (0.10, 0.15, 0.20), 0.08, 2.0, 20
LXYZ, RXYZ = slice(0, 3), slice(7, 10)
CATS = ("false_reach", "base_only", "blocked", "intermediate")


def episode(f, ref):
    z = np.load(f, allow_pickle=True); pose, act, dist = z["pose"], z["act"], z["dist"]
    d = np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]); yaw = pose[:, 2]
    moving = (np.linalg.norm(d, axis=1) / DT >= MOVE_V) | (np.abs(np.angle(np.exp(1j * np.diff(yaw, prepend=yaw[:1])))) / DT >= MOVE_W)
    dev = np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))
    cmd = (np.abs(act[:, 16]) >= CMD_V) | (np.abs(act[:, 17]) >= CMD_W)
    far = (dist > FAR_M) & (np.arange(len(dist)) >= TUCK_SKIP)
    stall = far & ~moving
    out = {"far_steps": int(far.sum()), "stall_steps": int(stall.sum())}
    for T in THRESHOLDS:
        out[f"{T:.2f}"] = {"false_reach": int((stall & (dev > T)).sum()),
                           "base_only": int((stall & (dev <= TUCK_MAX) & ~cmd).sum()),
                           "blocked": int((stall & (dev <= TUCK_MAX) & cmd).sum()),
                           "intermediate": int((stall & (dev > TUCK_MAX) & (dev <= T)).sum())}
    near = (dist <= FAR_M) & ~moving
    out["near_stall_steps"] = int(near.sum())
    return out


def summarize(eps, rng):
    S = np.array([e["stall_steps"] for e in eps], float)
    res = {"episodes": len(eps), "far_steps": int(sum(e["far_steps"] for e in eps)), "stall_steps": int(S.sum()),
           "stall_share_of_far_time": round(float(S.sum() / max(1, sum(e["far_steps"] for e in eps))), 4),
           "near_target_stall_steps": int(sum(e["near_stall_steps"] for e in eps)), "by_threshold": {}}
    for T in (f"{t:.2f}" for t in THRESHOLDS):
        C = {c: np.array([e[T][c] for e in eps], float) for c in CATS}
        share = {}
        for c in CATS:
            boot = []
            for _ in range(5000):
                i = rng.integers(0, len(eps), len(eps))
                boot.append(C[c][i].sum() / max(1.0, S[i].sum()))
            share[c] = {"share": round(float(C[c].sum() / max(1.0, S.sum())), 4),
                        "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}
        mostly = sum(1 for k in range(len(eps)) if S[k] > 0 and C["false_reach"][k] / S[k] > 0.5)
        res["by_threshold"][T] = {"shares": share, "episodes_with_stall": int((S > 0).sum()),
                                  "episodes_mostly_false_reach": mostly}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="LABEL=DIR")
    ap.add_argument("--expert-pose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    rng = np.random.default_rng(0)
    rep = {"definition": __doc__, "runs": {}}
    for spec in a.traces:
        lab, d = spec.split("=", 1)
        eps = [episode(f, ref) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
        rep["runs"][lab] = r = summarize(eps, rng)
        b = r["by_threshold"]
        print(f"{lab:16s} eps {r['episodes']} stall {r['stall_steps']}/{r['far_steps']} far steps ({r['stall_share_of_far_time']}) | "
              + " | ".join(f"T={T}: " + " ".join(f"{c} {x['shares'][c]['share']:.3f}{x['shares'][c]['ci95']}" for c in CATS)
                           + f" mostly_FR {x['episodes_mostly_false_reach']}/{x['episodes_with_stall']}" for T, x in b.items()), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
