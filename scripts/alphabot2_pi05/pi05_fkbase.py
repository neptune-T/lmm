import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/nix/yanjun/zts/alphabot2/lmm_repo")
from lmm.embodiments.alphabot2 import pi05_contract as P
D = "/nix/yanjun/zts/alphabot2/hf/dataset"
df = pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
S = np.stack(df["observation.state"].to_numpy()).astype(float); A = np.stack(df["action"].to_numpy()).astype(float)
TI = df["task_index"].to_numpy()
out = {}
for side, ss, sp in (("left", slice(0, 7), slice(0, 3)), ("right", slice(7, 14), slice(7, 10))):
    qv = S[:, ss] * P.DATASET_JOINT_SIGN[side]
    xyz = np.stack([P.action_from_vendor_joints(side, q)[0] for q in qv])
    e = np.linalg.norm(xyz - A[:, sp], axis=1) * 1000
    out[side] = {"current_pose_vs_target_mm_median": round(float(np.median(e)), 1),
                 "per_task": {int(t): round(float(np.median(e[TI == t])), 1) for t in np.unique(TI)}}
print(json.dumps(out))
