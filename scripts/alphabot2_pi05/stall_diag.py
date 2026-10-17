#!/usr/bin/env python3
"""NAV stall diagnosis from closed-loop eval traces (eval_policy.py TRACE_DIR: per-step base pose, commanded
18-D action, IK flag, distance to the target object). Read-only.

Per step (10 Hz): actual forward speed / yaw rate from the pose, commanded v / omega from the action.
  moving       actual |v| >= 0.03 m/s or |omega| >= 0.05 rad/s
  policy_zero  not moving and commanded |v| < 0.03 and |omega| < 0.05    (the policy asks to stand still)
  blocked      not moving although commanded |v| >= 0.03 or |omega| >= 0.05 (executor / collision / navmesh)
  manip_like   commanded arm target off the expert's NAV (tucked) pose by > ARM_DEV_M on either arm, or a
               gripper commanded closed (> 0.5), while the target is > FAR_M away
The tucked reference is the median commanded arm position over the expert's navigate_* frames in the
training sources (per arm), so "off the NAV pose" means off what the expert does while driving.
Stall onset: first run of >= ONSET_STEPS consecutive non-moving steps; reports distance, causes and arm state
inside it and in the 50 steps before it. Everything is binned by distance to target.
    stall_diag.py --traces A_val=/runs/v2/eval_A/traces_val_pass0 ... --expert-manifest M.json --out diag.json
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

DT, MOVE_V, MOVE_W, CMD_V, CMD_W = 0.1, 0.03, 0.05, 0.03, 0.05
ARM_DEV_M, FAR_M, ONSET_STEPS = 0.05, 1.5, 100
BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]
LXYZ, RXYZ, GRIP, V, W = slice(0, 3), slice(7, 10), slice(14, 16), 16, 17


def expert_nav_pose(manifest, collect, n=40):
    m = json.load(open(manifest))
    rows = {}
    for f in glob.glob(f"{collect}/results*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("accepted") and r.get("dir"):
                rows[str(r["episode_id"])] = r["dir"]
    L, R = [], []
    for u in list(m["train"])[:n]:
        d = Path(rows[u])
        a, st = np.load(d / "lowdim.npz")["action"], [str(x) for x in np.load(d / "metadata.npz")["stage"]]
        nav = np.array([s.startswith("navigate") for s in st])
        L.append(a[nav][:, LXYZ]); R.append(a[nav][:, RXYZ])
    L, R = np.concatenate(L), np.concatenate(R)
    return {"left": np.median(L, 0), "right": np.median(R, 0),
            "left_spread_p95_m": float(np.percentile(np.linalg.norm(L - np.median(L, 0), axis=1), 95)),
            "right_spread_p95_m": float(np.percentile(np.linalg.norm(R - np.median(R, 0), axis=1), 95)), "n_frames": len(L)}


def per_step(z, ref):
    pose, act, dist, ik = z["pose"], z["act"], z["dist"], z["ik"]
    d = np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2])
    yaw = pose[:, 2]
    v_act = np.linalg.norm(d, axis=1) / DT
    w_act = np.abs(np.angle(np.exp(1j * np.diff(yaw, prepend=yaw[:1])))) / DT
    v_cmd, w_cmd = act[:, V], act[:, W]
    moving = (v_act >= MOVE_V) | (w_act >= MOVE_W)
    cmd_move = (np.abs(v_cmd) >= CMD_V) | (np.abs(w_cmd) >= CMD_W)
    arm_dev = np.maximum(np.linalg.norm(act[:, LXYZ] - ref["left"], axis=1), np.linalg.norm(act[:, RXYZ] - ref["right"], axis=1))
    grip = act[:, GRIP].max(1)
    return {"dist": dist, "v_act": v_act, "w_act": w_act, "v_cmd": v_cmd, "w_cmd": w_cmd, "moving": moving,
            "policy_zero": ~moving & ~cmd_move, "blocked": ~moving & cmd_move, "arm_dev": arm_dev, "grip": grip,
            "manip_like": ((arm_dev > ARM_DEV_M) | (grip > 0.5)) & (dist > FAR_M), "ik_bad": ~ik.astype(bool)}


def onset(s):
    still = ~s["moving"]
    run = 0
    for t, x in enumerate(still):
        run = run + 1 if x else 0
        if run >= ONSET_STEPS:
            t0 = t - ONSET_STEPS + 1
            seg = slice(t0, len(still))
            end = t0 + int(np.argmax(s["moving"][t0:])) if s["moving"][t0:].any() else len(still)
            seg, pre = slice(t0, end), slice(max(0, t0 - 50), t0)
            win = np.flatnonzero(s["arm_dev"][max(0, t0 - 300):t0 + ONSET_STEPS] > ARM_DEV_M)   # arm leaves the NAV pose
            arm_lead = (t0 - (max(0, t0 - 300) + int(win[0]))) if len(win) else None              # > 0: before the stall
            return {"step": t0, "dist_m": round(float(s["dist"][t0]), 2), "length": end - t0, "recovers": end < len(still),
                    "arm_lead_steps": arm_lead,
                    "in_stall": {"policy_zero": round(float(s["policy_zero"][seg].mean()), 3), "blocked": round(float(s["blocked"][seg].mean()), 3),
                                 "v_cmd_med": round(float(np.median(s["v_cmd"][seg])), 3), "w_cmd_abs_med": round(float(np.median(np.abs(s["w_cmd"][seg]))), 3),
                                 "arm_dev_med_m": round(float(np.median(s["arm_dev"][seg])), 3), "grip_closed": round(float((s["grip"][seg] > 0.5).mean()), 3),
                                 "ik_bad": round(float(s["ik_bad"][seg].mean()), 3)},
                    "before": {"v_cmd_med": round(float(np.median(s["v_cmd"][pre])), 3) if pre.stop > pre.start else None,
                               "arm_dev_med_m": round(float(np.median(s["arm_dev"][pre])), 3) if pre.stop > pre.start else None,
                               "dist_drop_m": round(float(s["dist"][pre.start] - s["dist"][t0]), 2) if pre.stop > pre.start else None}}
    return None


def summarize(steps, onsets):
    cat = {k: np.concatenate([s[k] for s in steps]) for k in steps[0]}
    out = {"steps": int(len(cat["dist"])), "bins": {}}
    for lo, hi in BINS:
        m = (cat["dist"] >= lo) & (cat["dist"] < hi)
        if not m.any():
            continue
        f = lambda k: round(float(cat[k][m].mean()), 3)
        out["bins"][f"{lo}-{hi}m"] = {
            "step_share": round(float(m.mean()), 3), "moving": f("moving"), "policy_zero": f("policy_zero"), "blocked": f("blocked"),
            "manip_like": f("manip_like"), "ik_bad": f("ik_bad"),
            "v_cmd_mean": f("v_cmd"), "v_act_mean": f("v_act"), "w_cmd_abs_mean": round(float(np.abs(cat["w_cmd"][m]).mean()), 3),
            "w_act_mean": f("w_act"), "arm_dev_med_m": round(float(np.median(cat["arm_dev"][m])), 3),
            "grip_closed": round(float((cat["grip"][m] > 0.5).mean()), 3),
            "v_act_over_cmd_when_cmd_v>0.1": round(float(cat["v_act"][m & (cat["v_cmd"] > 0.1)].mean() / cat["v_cmd"][m & (cat["v_cmd"] > 0.1)].mean()), 3)
            if (m & (cat["v_cmd"] > 0.1)).any() else None}
    on = [o for o in onsets if o]
    out["episodes"] = len(onsets)
    out["stall_onset"] = {"episodes_with_onset": len(on), "dist_m_med": round(float(np.median([o["dist_m"] for o in on])), 2) if on else None,
                          "step_med": int(np.median([o["step"] for o in on])) if on else None,
                          "recovers": sum(o["recovers"] for o in on),
                          "dominant_cause": {c: sum(1 for o in on if max(("policy_zero", "blocked"), key=lambda k: o["in_stall"][k]) == c)
                                             for c in ("policy_zero", "blocked")},
                          "arm_off_nav_pose_in_stall": sum(o["in_stall"]["arm_dev_med_m"] > ARM_DEV_M for o in on),
                          "grip_closed_in_stall": sum(o["in_stall"]["grip_closed"] > 0.5 for o in on),
                          "arm_off_before_stall": sum(o["arm_lead_steps"] is not None and o["arm_lead_steps"] > 0 for o in on),
                          "arm_off_only_after_stall": sum(o["arm_lead_steps"] is not None and o["arm_lead_steps"] <= 0 for o in on),
                          "arm_never_off": sum(o["arm_lead_steps"] is None for o in on),
                          "arm_lead_steps_med": int(np.median([o["arm_lead_steps"] for o in on if o["arm_lead_steps"] is not None]))
                          if any(o["arm_lead_steps"] is not None for o in on) else None,
                          "onset_dist_bins": {f"{lo}-{hi}m": sum(lo <= o["dist_m"] < hi for o in on) for lo, hi in BINS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="LABEL=DIR")
    ap.add_argument("--expert-manifest", required=True)
    ap.add_argument("--collect", default="/workspace/runtime/runs/collect_clean_v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = expert_nav_pose(a.expert_manifest, a.collect)
    rep = {"thresholds": {"move_v": MOVE_V, "move_w": MOVE_W, "cmd_v": CMD_V, "cmd_w": CMD_W, "arm_dev_m": ARM_DEV_M,
                          "far_m": FAR_M, "onset_steps": ONSET_STEPS, "dt": DT},
           "expert_nav_pose": {k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v) for k, v in ref.items()}, "runs": {}}
    for spec in a.traces:
        lab, d = spec.split("=", 1)
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        steps, ons, per_ep = [], [], {}
        for f in files:
            z = np.load(f, allow_pickle=True)
            s = per_step(z, ref)
            steps.append(s); o = onset(s); ons.append(o)
            per_ep[Path(f).stem] = {"start_dist": round(float(s["dist"][0]), 2), "min_dist": round(float(s["dist"].min()), 2),
                                    "moving": round(float(s["moving"].mean()), 3), "policy_zero": round(float(s["policy_zero"].mean()), 3),
                                    "blocked": round(float(s["blocked"].mean()), 3), "manip_like": round(float(s["manip_like"].mean()), 3),
                                    "onset": o, "funnel": json.loads(str(z["funnel"]))}
        rep["runs"][lab] = {"dir": d, **summarize(steps, ons), "per_episode": per_ep}
        r = rep["runs"][lab]
        print(f"== {lab}: {r['episodes']} episodes, {r['steps']} steps; onset {json.dumps(r['stall_onset'])}", flush=True)
        for b, x in r["bins"].items():
            print(f"   {b:6s} {json.dumps(x)}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
