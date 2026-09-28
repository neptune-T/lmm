import json, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
exec(open("/nix/yanjun/zts/alphabot2/hf/pi05_joint_conv.py").read().split("rng = np.random.default_rng(0)")[0])
E = df["episode_index"].to_numpy(); TI = df["task_index"].to_numpy()
rng = np.random.default_rng(5)
CFG = {"right": ((1, 1, -1, 1, -1, 1, -1), slice(7, 14), slice(7, 10), slice(10, 14)),
       "left": ((1, 1, 1, 1, 1, 1, 1), slice(0, 7), slice(0, 3), slice(3, 7))}
for side, (s, ss, sp, sq) in CFG.items():
    s = np.array(s, float)
    for k in (0, 2, 4):
        ia, ib = [], []
        for e in np.unique(E):
            ii = np.nonzero(E == e)[0]
            if len(ii) > k: ia.extend(ii[:len(ii) - k]); ib.extend(ii[k:])
        ia, ib = np.array(ia), np.array(ib)
        sel = rng.choice(len(ia), 1500, replace=False); ia, ib = ia[sel], ib[sel]
        Pa = A[ia, sp]; Ra = Rot.from_quat(A[ia, sq]).as_matrix(); Qd = S[ib, ss]
        def mdl(x):
            M = fk_batch(side, Qd * s + x[12:19])
            Rb = Rot.from_rotvec(x[0:3]).as_matrix(); Rt = Rot.from_rotvec(x[6:9]).as_matrix()
            R = np.einsum("ij,njk,kl->nil", Rb, M[:, :3, :3], Rt)
            p = (Rb @ (M[:, :3, 3] + M[:, :3, :3] @ x[9:12]).T).T + x[3:6]
            return R, p
        def resid(x):
            R, p = mdl(x)
            return np.concatenate([(p - Pa).ravel(), (Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec() * 0.2).ravel()])
        best = None
        for s0 in Rot.random(8, random_state=7).as_rotvec():
            for s1 in Rot.random(3, random_state=8).as_rotvec():
                x0 = np.r_[s0, 0, 0, -0.3, s1, 0, 0, 0.1, np.zeros(7)]
                lb = np.r_[-np.inf * np.ones(12), -0.6 * np.ones(7)]; ub = -lb
                r = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=200)
                if best is None or r.cost < best.cost: best = r
        R, p = mdl(best.x)
        pe = np.linalg.norm(p - Pa, axis=1) * 1000
        re = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec(), axis=1))
        pct = lambda v: [round(float(np.percentile(v, q)), 2) for q in (10, 25, 50, 75, 90)]
        out = {"side": side, "k": k, "pos_mm_p10..90": pct(pe), "rot_deg_p10..90": pct(re),
               "joint_offsets_rad": np.round(best.x[12:19], 4).tolist(),
               "Rb": np.round(Rot.from_rotvec(best.x[0:3]).as_matrix(), 4).tolist(), "tb": np.round(best.x[3:6], 4).tolist(),
               "Rt": np.round(Rot.from_rotvec(best.x[6:9]).as_matrix(), 4).tolist(), "tt": np.round(best.x[9:12], 4).tolist(),
               "per_task_median_mm_deg": {int(t): [round(float(np.median(pe[TI[ia] == t])), 1), round(float(np.median(re[TI[ia] == t])), 2)] for t in np.unique(TI[ia])}}
        O[f"{side}_k{k}"] = out
        print(json.dumps(out), flush=True)
print("###JSON###" + json.dumps(O))
