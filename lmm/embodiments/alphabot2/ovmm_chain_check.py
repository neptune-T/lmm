"""Lowering-chain check of recorded AlphaBot 2 OVMM demonstrations (numpy only; interface gate + tests).

Never compares joint-space state dims with Cartesian action dims. Per episode:
  lowering     replaying CartesianExecutor over the recorded CartesianMobileAction18 labels reproduces the
               recorded executed JointMobileAction18 (arms, grippers) and passes (v, omega) through;
  tracking     the executed arm joint targets are the next Habitat state's arm joints;
  FK semantic  each arm's ACTUAL next-state pose (FK of its joints) is that arm's Cartesian label
               (within the executor's IK tolerance), so a left/right or slot permutation fails;
  gripper      open commands -> open next state, close commands -> closed/holding next state;
  base         the LABEL (v, omega) vs the base's actual forward displacement / yaw change over 0.1 s
               (sign and magnitude; navmesh sliding allowed for in the tolerance).
"""
import numpy as np

from . import cartesian as CE

def lowering_chain(low, md, dt=0.1):
    """CartesianMobileAction18 -> CartesianExecutor -> executed JointMobileAction18 -> next Habitat state."""
    cart, j18, pr = low["action"], low["joint18"], low["proprio"]
    T = len(cart)
    ex = CE.CartesianExecutor.from_proprio(pr[0])                  # the expert's executor starts from proprio[0]
    lowered = np.stack([ex.lower(cart[t])[0] for t in range(T)])
    arms = np.r_[0:7, 8:15]
    low_err = float(np.abs(lowered[:, arms] - j18[:, arms]).max())
    grip_err = float(np.abs(lowered[:, [7, 15]] - j18[:, [7, 15]]).max())
    base_err = max(float(np.abs(lowered[:, 16:18] - cart[:, 16:18]).max()), float(np.abs(lowered[:, 16:18] - j18[:, 16:18]).max()))
    lowering = {"arm_joint_max_abs_rad": low_err, "gripper_max_abs": grip_err, "base_passthrough_max_abs": base_err,
                "pass": low_err < 1e-6 and grip_err < 1e-6 and base_err < 1e-9}
    # executed joint target at t -> arm joints of the next state (kinematic arms; limits may clamp)
    d = np.abs(pr[1:][:, arms] - j18[:-1][:, arms]).max(axis=1)
    tracking = {"fraction_within_1e-4": float(np.mean(d < 1e-4)), "max_abs_rad": float(d.max()),
                "pass": float(np.mean(d < 1e-4)) >= 0.99}
    # semantic link: each arm's ACTUAL next-state pose (FK of its joints) is that arm's Cartesian label
    fk = {"left": [], "right": []}
    prev = {"left": None, "right": None}
    for t in range(T - 1):
        c_next = CE.cartesian_from_joint_action(np.r_[pr[t + 1], 0.0, 0.0], prev)
        for side, xs, qs in (("left", slice(0, 3), slice(3, 7)), ("right", slice(7, 10), slice(10, 14))):
            pos = float(np.linalg.norm(c_next[xs] - cart[t][xs]))
            qa, qb = c_next[qs] / np.linalg.norm(c_next[qs]), cart[t][qs] / np.linalg.norm(cart[t][qs])
            ang = float(np.degrees(2 * np.arccos(min(1.0, abs(float(qa @ qb))))))
            fk[side].append((pos, ang))
    fkc = {}
    for side, v in fk.items():
        a_ = np.asarray(v)
        fkc[side] = {"within_5mm_2deg": float(np.mean((a_[:, 0] <= 0.005) & (a_[:, 1] <= 2.0))),
                     "pos_err_median_m": float(np.median(a_[:, 0])), "ang_err_median_deg": float(np.median(a_[:, 1]))}
    fk_pass = all(v["within_5mm_2deg"] >= 0.99 for v in fkc.values())
    # gripper: command (Cartesian dims 14/15 -> executed 7/15) -> next-state closed fraction (proprio 7/15)
    g = {}
    for side, ci, pi in (("left", 14, 7), ("right", 15, 15)):
        cmd, nxt = j18[:-1, pi], pr[1:, pi]
        steady = np.r_[False, False, cmd[2:] == cmd[1:-1]] & np.r_[False, cmd[1:] == cmd[:-1]]
        op, cl = steady & (cmd <= 0.05), steady & (cmd >= 0.95)
        g[side] = {"cart_equals_executed": bool(np.allclose(cart[:-1, ci], cmd, atol=1e-9) or np.allclose(lowered[:-1, pi], cmd, atol=1e-9)),
                   "open_steps": int(op.sum()), "open_next_max_closed_fraction": float(nxt[op].max()) if op.any() else None,
                   "close_steps": int(cl.sum()), "close_next_min_closed_fraction": float(nxt[cl].min()) if cl.any() else None}
        g[side]["pass"] = ((not op.any() or g[side]["open_next_max_closed_fraction"] <= 0.05)
                           and (not cl.any() or g[side]["close_next_min_closed_fraction"] >= 0.15))
    gripper = {**g, "pass": all(v["pass"] for v in g.values()) and any(v["close_steps"] for v in g.values())}
    # base: commanded (v, omega) vs actual forward displacement / yaw change over the 0.1 s step
    b = np.asarray(md["base_xz_yaw"], float)
    v, w = cart[:-1, 16], cart[:-1, 17]                               # the TRAINING LABEL vs what the base did
    dxz = b[1:, :2] - b[:-1, :2]
    fwd = np.c_[np.cos(b[:-1, 2]), -np.sin(b[:-1, 2])]               # pose2d: yaw about +y, 0 = +x, forward (cos, -sin)
    adv = np.sum(dxz * fwd, axis=1)
    dyaw = (b[1:, 2] - b[:-1, 2] + np.pi) % (2 * np.pi) - np.pi
    mv, tn = np.abs(v) > 0.05, np.abs(w) > 0.05
    rv = adv[mv] / (v[mv] * dt) if mv.any() else np.zeros(0)
    rw = dyaw[tn] / (w[tn] * dt) if tn.any() else np.zeros(0)
    base = {"moving_steps": int(mv.sum()), "v_sign_agreement": float(np.mean(rv > 0)) if mv.any() else None,
            "v_ratio_median": float(np.median(rv)) if mv.any() else None,
            "v_ratio_within_0.5_1.1": float(np.mean((rv >= 0.5) & (rv <= 1.1))) if mv.any() else None,
            "turning_steps": int(tn.sum()), "omega_sign_agreement": float(np.mean(rw > 0)) if tn.any() else None,
            "omega_ratio_median": float(np.median(rw)) if tn.any() else None,
            "omega_ratio_within_0.8_1.2": float(np.mean((rw >= 0.8) & (rw <= 1.2))) if tn.any() else None}
    base["pass"] = bool(mv.any() and tn.any() and base["v_sign_agreement"] >= 0.98 and base["omega_sign_agreement"] >= 0.98
                        and base["v_ratio_within_0.5_1.1"] >= 0.9 and base["omega_ratio_within_0.8_1.2"] >= 0.9)
    return {"steps": int(T), "lowering": lowering, "joint_tracking": tracking,
            "fk_next_state_matches_label": {**fkc, "pass": fk_pass}, "gripper": gripper, "base": base}
