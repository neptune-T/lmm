"""Identify the dataset's per-arm joint convention relative to the xacro.

q_xacro = s * q_data + o. Scored with a frame-free invariant: the rotation
angle between two samples' EE orientations must equal the angle between their
FK orientations, whatever the (possibly mirrored) base and tool frames are.
"""
import itertools, json, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/nix/yanjun/zts/alphabot2/lmm_repo/scripts/alphabot2_isaac")
from xacro_fk import XacroArms, _tf
XF = XacroArms("/nix/yanjun/zts/alphabot2/deps/robot_descriptions/humanoid/Ai2Robotics/ai2_bot2_description/xacro/components")
D = "/nix/yanjun/zts/alphabot2/hf/dataset"
df = pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
A = np.stack(df["action"].to_numpy()).astype(np.float64)
O = {}


def chain(side):
    J = XF.arm[side]
    pre = np.eye(4)
    for n in ("body_joint1", "body_joint2", "body_joint3", "body_joint4"):
        pre = pre @ _tf(XF.body[n]["xyz"], XF.body[n]["rpy"])
    pre = pre @ _tf(J["${prefix}joint0"]["xyz"], J["${prefix}joint0"]["rpy"])
    orig = [_tf(J[f"${{prefix}}joint{k}"]["xyz"], J[f"${{prefix}}joint{k}"]["rpy"]) for k in range(1, 8)]
    axes = [np.asarray(J[f"${{prefix}}joint{k}"]["axis"], float) for k in range(1, 8)]
    post = _tf(J["${prefix}joint8"]["xyz"], J["${prefix}joint8"]["rpy"]) @ _tf([0, 0, 0.0695], [0, 0, 0]) @ _tf([0, 0, 0], [0, 0, 1.5708])
    return pre, orig, axes, post


CH = {s: chain(s) for s in ("left", "right")}


def fk_batch(side, Q):
    pre, orig, axes, post = CH[side]
    N = len(Q)
    M = np.broadcast_to(pre, (N, 4, 4)).copy()
    for k in range(7):
        a = axes[k] / np.linalg.norm(axes[k])
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        th = Q[:, k]
        R = np.eye(3)[None] + np.sin(th)[:, None, None] * K[None] + (1 - np.cos(th))[:, None, None] * (K @ K)[None]
        T = np.zeros((N, 4, 4)); T[:, :3, :3] = R; T[:, 3, 3] = 1
        M = M @ orig[k] @ T
    return M @ post


def angles(Ra, pairs):
    rel = np.einsum("nji,njk->nik", Ra[pairs[:, 0]], Ra[pairs[:, 1]])
    return np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))


rng = np.random.default_rng(0)
for side in ("left", "right"):
    qs = rng.uniform(-1, 1, (3, 7))
    ref = np.stack([XF.fk(side, q) for q in qs])
    assert np.abs(fk_batch(side, qs) - ref).max() < 1e-9, "batch FK mismatch"
O["batch_fk_check"] = "ok"
for side, ss, sq in (("left", slice(0, 7), slice(3, 7)), ("right", slice(7, 14), slice(10, 14))):
    # prefer frames where the arm actually moves
    idx = rng.choice(len(S), 400, replace=False)
    Qd, Ra = S[idx, ss], Rot.from_quat(A[idx, sq]).as_matrix()
    pairs = rng.integers(0, len(idx), size=(1500, 2))
    target = angles(Ra, pairs)
    O[f"{side}_pair_angle_spread_deg"] = round(float(np.degrees(target).std()), 2)

    def resid(o, s):
        M = fk_batch(side, Qd * s + o)
        return angles(M[:, :3, :3], pairs) - target

    results = []
    for s in itertools.product((1.0, -1.0), repeat=7):
        s = np.array(s)
        best = None
        for o0 in [np.zeros(7)] + [rng.uniform(-math.pi, math.pi, 7) for _ in range(3)]:
            r = least_squares(resid, o0, args=(s,), max_nfev=150)
            if best is None or r.cost < best.cost: best = r
        rms = math.degrees(math.sqrt(2 * best.cost / len(target)))
        results.append((rms, s.astype(int).tolist(), [round(float(((v + math.pi) % (2 * math.pi)) - math.pi), 4) for v in best.x]))
    results.sort(key=lambda r: r[0])
    O[f"{side}_top8"] = [{"rms_deg": round(r[0], 3), "sign": r[1], "offset": r[2],
                          "offset_over_pi2": [round(v / (math.pi / 2), 3) for v in r[2]]} for r in results[:8]]
    ident = [r for r in results if r[1] == [1] * 7][0]
    O[f"{side}_identity_sign_best"] = {"rms_deg": round(ident[0], 3), "offset": ident[2]}
    print(side, O[f"{side}_top8"][:3], flush=True)
print("###JSON###" + json.dumps(O))
