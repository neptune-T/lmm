"""Robust full-model fit per arm: a = Tb * FK(s*q+o) * Tt, soft-L1 loss; residual percentiles."""
import json, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
exec(open("/nix/yanjun/zts/alphabot2/hf/pi05_joint_conv.py").read().split("rng = np.random.default_rng(0)")[0])
E = df["episode_index"].to_numpy(); TI = df["task_index"].to_numpy()
rng = np.random.default_rng(5)
SIGNS = {"identity": np.ones(7)}
def model(x, side, Q):
    Rb = Rot.from_rotvec(x[0:3]).as_matrix(); tb = x[3:6]; Rt = Rot.from_rotvec(x[6:9]).as_matrix(); tt = x[9:12]
    M = fk_batch(side, Q)
    R = np.einsum("ij,njk,kl->nil", Rb, M[:, :3, :3], Rt)
    p = (Rb @ (M[:, :3, 3] + M[:, :3, :3] @ tt).T).T + tb
    return R, p
for side, ss, sp, sq in (("left", slice(0, 7), slice(0, 3), slice(3, 7)), ("right", slice(7, 14), slice(7, 10), slice(10, 14))):
    for k in (0, 2):
        ia, ib = [], []
        for e in np.unique(E):
            ii = np.nonzero(E == e)[0]
            if len(ii) > k: ia.extend(ii[:len(ii) - k]); ib.extend(ii[k:])
        ia, ib = np.array(ia), np.array(ib)
        sel = rng.choice(len(ia), 2000, replace=False); ia, ib = ia[sel], ib[sel]
        Q = S[ib, ss]; Pa = A[ia, sp]; Ra = Rot.from_quat(A[ia, sq]).as_matrix()
        def resid(x):
            R, p = model(x, side, Q)
            er = Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec() * 0.2   # 1 rad ~ 0.2 m
            return np.concatenate([(p - Pa).ravel(), er.ravel()])
        best = None
        for s0 in Rot.random(10, random_state=7).as_rotvec():
            for s1 in Rot.random(3, random_state=8).as_rotvec():
                x0 = np.r_[s0, 0.0, 0.0, -0.3, s1, 0.0, 0.0, 0.1]
                r = least_squares(resid, x0, loss="soft_l1", f_scale=0.01, max_nfev=150)
                if best is None or r.cost < best.cost: best = r
        R, p = model(best.x, side, Q)
        pe = np.linalg.norm(p - Pa, axis=1) * 1000
        re = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec(), axis=1))
        pct = lambda v: [round(float(np.percentile(v, q)), 2) for q in (10, 25, 50, 75, 90)]
        per_task = {int(t): [round(float(np.median(pe[TI[ia] == t])), 1), round(float(np.median(re[TI[ia] == t])), 2)] for t in np.unique(TI[ia])}
        out = {"k": k, "pos_err_mm_p10_25_50_75_90": pct(pe), "rot_err_deg_p10_25_50_75_90": pct(re),
               "frac_pos<10mm_and_rot<3deg": round(float(np.mean((pe < 10) & (re < 3))), 3),
               "per_task_median_[mm,deg]": per_task,
               "Rb_rotvec_deg": np.round(np.degrees(best.x[0:3]), 2).tolist(), "tb": np.round(best.x[3:6], 4).tolist(),
               "Rt_matrix": np.round(Rot.from_rotvec(best.x[6:9]).as_matrix(), 3).tolist(), "tt": np.round(best.x[9:12], 4).tolist()}
        O[f"{side}_k{k}"] = out
        print(side, json.dumps(out), flush=True)
print("###JSON###" + json.dumps(O))
