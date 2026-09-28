"""Robust fit of named right-arm joint-convention hypotheses: q_xacro = s*q_data + o."""
import json, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
exec(open("/nix/yanjun/zts/alphabot2/hf/pi05_joint_conv.py").read().split("rng = np.random.default_rng(0)")[0])
E = df["episode_index"].to_numpy(); TI = df["task_index"].to_numpy()
rng = np.random.default_rng(5)
P = math.pi
HYP = {
    "xacro_identity": ((1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0)),
    "copy_of_left_usd_like": ((1, 1, -1, 1, -1, -1, 1), (0, 0, P, 0, -P, 0, 0)),
    "same_axes_as_left_no_offset": ((1, 1, -1, -1, -1, -1, 1), (0, 0, 0, 0, 0, 0, 0)),
    "copy_alt": ((1, 1, -1, 1, -1, -1, 1), (0, 0, 0, 0, 0, 0, 0)),
}
def model(x, Q):
    Rb = Rot.from_rotvec(x[0:3]).as_matrix(); tb = x[3:6]; Rt = Rot.from_rotvec(x[6:9]).as_matrix(); tt = x[9:12]
    M = fk_batch("right", Q)
    return np.einsum("ij,njk,kl->nil", Rb, M[:, :3, :3], Rt), (Rb @ (M[:, :3, 3] + M[:, :3, :3] @ tt).T).T + tb
ia = rng.choice(len(S), 2000, replace=False)
Pa = A[ia, 7:10]; Ra = Rot.from_quat(A[ia, 10:14]).as_matrix()
for name, (s, o) in HYP.items():
    Q = S[ia, 7:14] * np.array(s, float) + np.array(o, float)
    def resid(x):
        R, p = model(x, Q)
        return np.concatenate([(p - Pa).ravel(), (Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec() * 0.2).ravel()])
    best = None
    for s0 in Rot.random(10, random_state=7).as_rotvec():
        for s1 in Rot.random(3, random_state=8).as_rotvec():
            r = least_squares(resid, np.r_[s0, 0, 0, -0.3, s1, 0, 0, 0.1], loss="soft_l1", f_scale=0.01, max_nfev=150)
            if best is None or r.cost < best.cost: best = r
    R, p = model(best.x, Q)
    pe = np.linalg.norm(p - Pa, axis=1) * 1000
    re = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec(), axis=1))
    pct = lambda v: [round(float(np.percentile(v, q)), 2) for q in (10, 25, 50, 75, 90)]
    out = {"pos_mm_p10..90": pct(pe), "rot_deg_p10..90": pct(re), "frac_good": round(float(np.mean((pe < 15) & (re < 3))), 3),
           "Rb_matrix": np.round(Rot.from_rotvec(best.x[0:3]).as_matrix(), 3).tolist(), "tb": np.round(best.x[3:6], 4).tolist(),
           "Rt_matrix": np.round(Rot.from_rotvec(best.x[6:9]).as_matrix(), 3).tolist(), "tt": np.round(best.x[9:12], 4).tolist()}
    O[name] = out
    print(name, json.dumps(out), flush=True)
print("###JSON###" + json.dumps(O))
