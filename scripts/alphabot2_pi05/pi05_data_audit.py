"""Quantitative audit of alphabot2/aibot2_grab_bag_combined_over_under.

Resolves from data, not names: action frame, absolute vs delta, time alignment,
tool offset, quaternion conventions, gripper polarity. Run in venv_pi05.
"""
import json, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/nix/yanjun/zts/alphabot2/lmm_repo/scripts/alphabot2_isaac")
from xacro_fk import XacroArms
XF = XacroArms("/nix/yanjun/zts/alphabot2/deps/robot_descriptions/humanoid/Ai2Robotics/ai2_bot2_description/xacro/components")
D = "/nix/yanjun/zts/alphabot2/hf/dataset"
O = {}
df = pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
ep = pd.read_parquet(f"{D}/meta/episodes/chunk-000/file-000.parquet")
tasks = pd.read_parquet(f"{D}/meta/tasks.parquet")
O["columns"] = list(df.columns)
O["n_frames"] = len(df)
O["tasks"] = tasks.reset_index().astype(str).to_dict(orient="records")
O["episode_meta_columns"] = list(ep.columns)[:40]
S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
A = np.stack(df["action"].to_numpy()).astype(np.float64)
E = df["episode_index"].to_numpy(); F = df["frame_index"].to_numpy(); T = df["timestamp"].to_numpy()
O["task_index_counts"] = df["task_index"].value_counts().to_dict()
# timing
dts = []
for e in np.unique(E):
    t = T[E == e]; dts.append(np.diff(t))
dts = np.concatenate(dts)
O["timestamp_dt"] = {"min": float(dts.min()), "max": float(dts.max()), "median": float(np.median(dts))}
ep_len = pd.Series(E).value_counts()
O["episode_len_frames"] = {"min": int(ep_len.min()), "median": float(ep_len.median()), "max": int(ep_len.max())}
# quaternions
for side, sl in (("left", slice(3, 7)), ("right", slice(10, 14))):
    q = A[:, sl]
    n = np.linalg.norm(q, axis=1)
    flips = 0
    for e in np.unique(E):
        qe = q[E == e]
        flips += int(np.sum(np.einsum("ij,ij->i", qe[1:], qe[:-1]) < 0))
    O[f"quat_{side}"] = {"norm_min": float(n.min()), "norm_max": float(n.max()), "qw_min": float(q[:, 3].min()),
                         "frac_qw_negative": float((q[:, 3] < 0).mean()), "sign_flips_between_consecutive_frames": flips}
# grippers
for side, ai, si in (("left", 14, 14), ("right", 15, 15)):
    a, s = A[:, ai], S[:, si]
    O[f"gripper_{side}"] = {"action_unique": sorted(set(np.round(a, 4).tolist()))[:10],
                            "state_min": float(s.min()), "state_max": float(s.max()),
                            "state_mean_when_action_0": float(s[a < 0.5].mean()) if (a < 0.5).any() else None,
                            "state_mean_when_action_1": float(s[a >= 0.5].mean()) if (a >= 0.5).any() else None}
    # lagged: state k frames after the action
    lag = {}
    for k in (0, 1, 3, 5, 10):
        m = np.zeros(len(a), bool); idx = []
        for e in np.unique(E):
            ii = np.nonzero(E == e)[0]
            if len(ii) > k: idx.extend(zip(ii[:len(ii) - k], ii[k:]))
        ia = np.array([p[0] for p in idx]); ib = np.array([p[1] for p in idx])
        if (a[ia] >= 0.5).any() and (a[ia] < 0.5).any():
            lag[k] = [round(float(s[ib][a[ia] < 0.5].mean()), 4), round(float(s[ib][a[ia] >= 0.5].mean()), 4)]
    O[f"gripper_{side}"]["state_mean_k_frames_later_[act0, act1]"] = lag
    # episode start values
    starts = np.array([np.nonzero(E == e)[0][0] for e in np.unique(E)])
    O[f"gripper_{side}"]["episode_start_action_mean"] = float(a[starts].mean())
    O[f"gripper_{side}"]["episode_start_state_mean"] = float(s[starts].mean())

# FK fit: action pose vs xacro FK of state at lag k
def fk_all(side, Q):
    return np.stack([XF.fk(side, q) for q in Q])

def fit(side, k):
    sl_s = slice(0, 7) if side == "left" else slice(7, 14)
    sl_p = slice(0, 3) if side == "left" else slice(7, 10)
    sl_q = slice(3, 7) if side == "left" else slice(10, 14)
    ia, ib = [], []
    for e in np.unique(E):
        ii = np.nonzero(E == e)[0]
        if len(ii) > k:
            ia.extend(ii[:len(ii) - k]); ib.extend(ii[k:])
    ia, ib = np.array(ia), np.array(ib)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ia), size=min(1500, len(ia)), replace=False)
    ia, ib = ia[sel], ib[sel]
    M = fk_all(side, S[ib, sl_s])
    Pa = A[ia, sl_p]; Ra = Rot.from_quat(A[ia, sl_q]).as_matrix()   # scipy quat = xyzw
    Rf = M[:, :3, :3]; Pf = M[:, :3, 3]
    # orientation: Ra = Rb Rf Rt
    def res_rot(x):
        Rb = Rot.from_rotvec(x[:3]).as_matrix(); Rt = Rot.from_rotvec(x[3:]).as_matrix()
        Rm = np.einsum("ij,njk,kl->nil", Rb, Rf, Rt)
        return Rot.from_matrix(np.einsum("nji,njk->nik", Ra, Rm)).as_rotvec().ravel()
    best = None
    for s0 in Rot.random(12, random_state=1).as_rotvec():
        for s1 in Rot.random(2, random_state=2).as_rotvec():
            r = least_squares(res_rot, np.r_[s0, s1], max_nfev=100)
            if best is None or r.cost < best.cost: best = r
    Rb = Rot.from_rotvec(best.x[:3]).as_matrix(); Rt = Rot.from_rotvec(best.x[3:]).as_matrix()
    rot_err = np.degrees(np.linalg.norm(best.fun.reshape(-1, 3), axis=1))
    # position: Pa = Rb (Pf + Rf t_tool) + t_b  -> linear in (t_b, t_tool)
    n = len(Pa)
    Amat = np.zeros((3 * n, 6)); b = np.zeros(3 * n)
    for i in range(n):
        Amat[3*i:3*i+3, :3] = np.eye(3)
        Amat[3*i:3*i+3, 3:] = Rb @ Rf[i]
        b[3*i:3*i+3] = Pa[i] - Rb @ Pf[i]
    sol, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    pos_err = np.linalg.norm((Amat @ sol - b).reshape(-1, 3), axis=1)
    return {"k": k, "n": n, "pos_err_mm": {"median": round(float(np.median(pos_err)) * 1000, 2), "p95": round(float(np.percentile(pos_err, 95)) * 1000, 2)},
            "rot_err_deg": {"median": round(float(np.median(rot_err)), 3), "p95": round(float(np.percentile(rot_err, 95)), 3)},
            "R_base_rotvec_deg": np.round(np.degrees(best.x[:3]), 2).tolist(), "R_base": np.round(Rb, 4).tolist(),
            "R_tool": np.round(Rt, 4).tolist(), "t_base_m": np.round(sol[:3], 4).tolist(), "t_tool_m": np.round(sol[3:], 4).tolist()}

for side in ("left", "right"):
    O[f"fk_fit_{side}"] = [fit(side, k) for k in (0, 1, 3)]

print("###JSON###" + json.dumps(O, default=str))
