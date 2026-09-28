"""In-distribution check: feed real dataset frames, compare predicted chunks to recorded future actions."""
import json, sys
import av, numpy as np, pandas as pd, torch
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/nix/yanjun/zts/alphabot2/hf")
from pi05_runtime import CAMS, Pi05Runtime

D = "/nix/yanjun/zts/alphabot2/hf/dataset"
df = pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
ep = pd.read_parquet(f"{D}/meta/episodes/chunk-000/file-000.parquet")
tasks = pd.read_parquet(f"{D}/meta/tasks.parquet")
tname = {int(r.task_index): i for i, r in tasks.iterrows()}
S = np.stack(df["observation.state"].to_numpy()); A = np.stack(df["action"].to_numpy())
E = df["episode_index"].to_numpy(); F = df["frame_index"].to_numpy(); TI = df["task_index"].to_numpy()

def frame(e, fi):
    r = ep[ep.episode_index == e].iloc[0]
    out = {}
    for c in CAMS:
        t = float(r[f"videos/observation.images.{c}/from_timestamp"]) + fi / 10.0
        cont = av.open(f"{D}/videos/observation.images.{c}/chunk-000/file-000.mp4"); st = cont.streams.video[0]
        cont.seek(int(max(0, t - 1.0) / st.time_base), stream=st, backward=True)
        for fr in cont.decode(st):
            if fr.time is not None and fr.time >= t - 0.049:
                out[c] = fr.to_ndarray(format="rgb24"); break
        cont.close()
    return out

rt = Pi05Runtime()
rng = np.random.default_rng(0)
lens = pd.Series(E).value_counts()
picks = []
for task in (0, 1, 3):
    eps = [e for e in lens.index if TI[E == e][0] == task and lens[e] >= 60]
    for e in rng.choice(eps, 4, replace=False):
        picks.append((int(e), int(rng.integers(5, lens[e] - 40))))
rows, lat = [], []
for e, fi in picks:
    idx = np.nonzero((E == e) & (F == fi))[0][0]
    ims = frame(e, fi)
    task = tname[int(TI[idx])]
    pred, dt = rt.infer(ims, S[idx], task, seed=0)
    lat.append(dt)
    H = 20
    gt = A[idx:idx + H]
    def perr(P, G, sl):
        return float(np.mean(np.linalg.norm(P[:, sl] - G[:, sl], axis=1)) * 1000)
    def rerr(P, G, sl):
        return float(np.mean(np.degrees((Rot.from_quat(P[:, sl]).inv() * Rot.from_quat(G[:, sl])).magnitude())))
    hold = np.repeat(A[idx:idx + 1], H, axis=0)
    qn = np.linalg.norm(pred[:, 10:14], axis=1)
    rows.append({"episode": e, "frame": fi, "task": task,
                 "right_pos_err_mm_pred": round(perr(pred[:H], gt, slice(7, 10)), 1),
                 "right_pos_err_mm_hold_baseline": round(perr(hold, gt, slice(7, 10)), 1),
                 "right_rot_err_deg_pred": round(rerr(pred[:H], gt, slice(10, 14)), 2),
                 "right_rot_err_deg_hold": round(rerr(hold, gt, slice(10, 14)), 2),
                 "left_pos_err_mm_pred": round(perr(pred[:H], gt, slice(0, 3)), 1),
                 "right_grip_pred_first10": np.round(pred[:10, 15], 2).tolist(), "right_grip_gt_first10": np.round(gt[:10, 15], 2).tolist(),
                 "left_grip_pred_max": round(float(np.abs(pred[:, 14]).max()), 3),
                 "pred_quat_norm_range": [round(float(qn.min()), 3), round(float(qn.max()), 3)],
                 "pred_qw_min": round(float(pred[:, 13].min()), 3)})
    print(json.dumps(rows[-1]), flush=True)
summ = {k: round(float(np.median([r[k] for r in rows])), 2) for k in rows[0] if k.endswith(("_pred", "_hold", "_baseline"))}
out = {"n": len(rows), "median": summ, "latency_s": {"first": round(lat[0], 3), "median_rest": round(float(np.median(lat[1:])), 3)},
       "rows": rows}
print("###JSON###" + json.dumps(out))
