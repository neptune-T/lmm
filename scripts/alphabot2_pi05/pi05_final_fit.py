"""Final frame fit: dataset joint conventions fixed (offsets 0), lag scan, shared vs per-arm base frame."""
import json, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
exec(open("/nix/yanjun/zts/alphabot2/hf/pi05_joint_conv.py").read().split("rng = np.random.default_rng(0)")[0])
E = df["episode_index"].to_numpy(); TI = df["task_index"].to_numpy()
rng = np.random.default_rng(9)
SIGN = {"left": np.ones(7), "right": np.array([1, 1, -1, 1, -1, 1, -1.0])}
SL = {"left": (slice(0, 7), slice(0, 3), slice(3, 7)), "right": (slice(7, 14), slice(7, 10), slice(10, 14))}
def pairs(k, n=1500, task=None):
    ia, ib = [], []
    for e in np.unique(E):
        ii = np.nonzero(E == e)[0]
        if task is not None and TI[ii[0]] != task: continue
        if len(ii) > k: ia.extend(ii[:len(ii) - k]); ib.extend(ii[k:])
    ia, ib = np.array(ia), np.array(ib)
    sel = rng.choice(len(ia), min(n, len(ia)), replace=False)
    return ia[sel], ib[sel]
def data(side, k, task=None):
    ss, sp, sq = SL[side]; ia, ib = pairs(k, task=task)
    M = fk_batch(side, S[ib, ss] * SIGN[side])
    return M, A[ia, sp], Rot.from_quat(A[ia, sq]).as_matrix(), TI[ia]
def pred(M, xb, xt):
    Rb = Rot.from_rotvec(xb[:3]).as_matrix(); Rt = Rot.from_rotvec(xt[:3]).as_matrix()
    R = np.einsum("ij,njk,kl->nil", Rb, M[:, :3, :3], Rt)
    p = (Rb @ (M[:, :3, 3] + M[:, :3, :3] @ xt[3:6]).T).T + xb[3:6]
    return R, p
def res(M, Pa, Ra, xb, xt):
    R, p = pred(M, xb, xt)
    return np.concatenate([(p - Pa).ravel(), (Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec() * 0.2).ravel()])
def stats(M, Pa, Ra, xb, xt):
    R, p = pred(M, xb, xt)
    pe = np.linalg.norm(p - Pa, axis=1) * 1000
    re = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("nji,njk->nik", Ra, R)).as_rotvec(), axis=1))
    return pe, re
def fit_sep(D):
    best = None
    for s0 in Rot.random(6, random_state=7).as_rotvec():
        for s1 in Rot.random(4, random_state=8).as_rotvec():
            r = least_squares(lambda x: res(*D[:3], x[:6], x[6:]), np.r_[s0, 0, 0, -0.4, s1, 0, 0, 0.12],
                              loss="soft_l1", f_scale=0.01, max_nfev=150)
            if best is None or r.cost < best.cost: best = r
    return best.x
pct = lambda v: [round(float(np.percentile(v, q)), 1) for q in (10, 25, 50, 75, 90)]
# lag scan, per arm
for side in ("left", "right"):
    for k in (2, 4, 6, 8, 10):
        D = data(side, k)
        x = fit_sep(D)
        pe, re = stats(*D[:3], x[:6], x[6:])
        O[f"lag_{side}_k{k}"] = {"pos_mm": pct(pe), "rot_deg": [round(float(np.percentile(re, q)), 2) for q in (10, 25, 50, 75, 90)]}
        print(side, k, O[f"lag_{side}_k{k}"], flush=True)
# shared base at k=4 on task 3 (pick & place) vs separate
k = 4
DL, DR = data("left", k, task=3), data("right", k, task=3)
xl, xr = fit_sep(DL), fit_sep(DR)
def res_shared(x):
    return np.concatenate([res(*DL[:3], x[:6], x[6:12]), res(*DR[:3], x[:6], x[12:18])])
best = None
for s0 in Rot.random(6, random_state=3).as_rotvec():
    x0 = np.r_[s0, xl[3:6], xl[6:12], xr[6:12]]
    r = least_squares(res_shared, x0, loss="soft_l1", f_scale=0.01, max_nfev=300)
    if best is None or r.cost < best.cost: best = r
for name, (D, xb, xt) in {"sep_left": (DL, xl[:6], xl[6:]), "sep_right": (DR, xr[:6], xr[6:]),
                         "shared_left": (DL, best.x[:6], best.x[6:12]), "shared_right": (DR, best.x[:6], best.x[12:18])}.items():
    pe, re = stats(*D[:3], xb, xt)
    O[f"task3_k4_{name}"] = {"pos_mm": pct(pe), "rot_deg": [round(float(np.percentile(re, q)), 2) for q in (10, 25, 50, 75, 90)],
                             "Rb": np.round(Rot.from_rotvec(xb[:3]).as_matrix(), 4).tolist(), "tb": np.round(xb[3:6], 4).tolist(),
                             "Rt": np.round(Rot.from_rotvec(xt[:3]).as_matrix(), 4).tolist(), "tt": np.round(xt[3:6], 4).tolist()}
    print(name, json.dumps(O[f"task3_k4_{name}"]), flush=True)
print("###JSON###" + json.dumps(O))
