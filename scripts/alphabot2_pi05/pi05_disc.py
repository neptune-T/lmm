import itertools, json, math, sys
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
exec(open("/nix/yanjun/zts/alphabot2/hf/pi05_joint_conv.py").read().split("rng = np.random.default_rng(0)")[0])
E = df["episode_index"].to_numpy()
rng = np.random.default_rng(11)
def within_pairs(n=3000):
    out = []
    eps = np.unique(E)
    while len(out) < n:
        e = rng.choice(eps); ii = np.nonzero(E == e)[0]
        if len(ii) >= 2: out.append(rng.choice(ii, 2, replace=False))
    return np.array(out)
res_all = {}
for side, ss, sq, fixed in (("right", slice(7, 14), slice(10, 14), {1: (1, 0), 3: (1, 0), 6: (-1, 0)}),
                            ("left", slice(0, 7), slice(3, 7), {1: (1, 0), 3: (1, 0), 6: (1, 0)})):
    pr = within_pairs()
    Ra = Rot.from_quat(A[:, sq]).as_matrix()
    rel = np.einsum("nji,njk->nik", Ra[pr[:, 0]], Ra[pr[:, 1]])
    target = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    Q1, Q2 = S[pr[:, 0], ss], S[pr[:, 1], ss]
    free = [0, 2, 4, 5]
    offs = [0.0, math.pi / 2, -math.pi / 2, math.pi]
    out = []
    for sg in itertools.product((1, -1), repeat=4):
        for of in itertools.product(offs, repeat=4):
            s = np.ones(7); o = np.zeros(7)
            for j, (a, b) in fixed.items(): s[j], o[j] = a, b
            for j, a, b in zip(free, sg, of): s[j], o[j] = a, b
            M1 = fk_batch(side, Q1 * s + o); M2 = fk_batch(side, Q2 * s + o)
            r = np.einsum("nji,njk->nik", M1[:, :3, :3], M2[:, :3, :3])
            ang = np.arccos(np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1))
            d = np.degrees(np.abs(ang - target))
            out.append((float(np.median(d)), float(np.mean(d < 2.0)), s.astype(int).tolist(), [round(v / (math.pi / 2)) for v in o]))
    out.sort(key=lambda r: r[0])
    res_all[side] = {"target_median_deg": round(float(np.degrees(np.median(target))), 2),
                     "top8": [{"median_abs_err_deg": round(a, 3), "frac_err<2deg": round(b, 3), "sign": c, "offset_pi2": d} for a, b, c, d in out[:8]],
                     "identity": [{"median_abs_err_deg": round(a, 3), "frac_err<2deg": round(b, 3)} for a, b, c, d in out if c == [1, 1, 1, 1, 1, 1, fixed[6][0]] and d == [0] * 7][:1]}
    print(side, json.dumps(res_all[side]), flush=True)
print("###JSON###" + json.dumps(res_all))
