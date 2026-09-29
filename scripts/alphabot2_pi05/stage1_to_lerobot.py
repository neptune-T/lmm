"""Stage-1 accepted episodes of ONE split -> a local LeRobot v3 dataset.

Separate datasets per split (train / val) so that lerobot's stats of the TRAIN
dataset are the only normalization statistics, and nothing held out can leak
into them. Only episodes that passed every quality gate (``episodes/``, meta
``accepted``) and whose config split matches are used; quarantined episodes
are never read.

Layout identical to the public AlphaBot checkpoint wherever it can be:
  observation.state      16-D, checkpoint convention (dataset_state_from_proprio)
  observation.images.*   camera_front <- cam_chest, camera_top <- cam_front (head),
                         camera_left <- cam_right_wrist, camera_right <- cam_left_wrist
  action                 18-D CartesianMobileAction18 (public 16 dims + v, omega)
  task                   the episode's language instruction
No privileged or stage feature is written. Images are stored lossless (PNG) so
training frames are bit-identical to what the policy sees at inference.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import numpy as np

REPO = os.environ.get("LMM_REPO", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402

CAM_MAP = {"camera_front": "cam_chest", "camera_top": "cam_front",
           "camera_left": "cam_right_wrist", "camera_right": "cam_left_wrist"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="collection OUT dirs (each has episodes/)")
    ap.add_argument("--split", required=True, choices=("train", "val"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--only", default=None, help="file with config_ids to include (e.g. an overfit subset)")
    ap.add_argument("--writer-threads", type=int, default=16)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    only = set(Path(args.only).read_text().split()) if args.only else None
    img = {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": list(P.STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (18,), "names": list(C.CARTESIAN_MOBILE_ACTION18_LABELS)},
        **{f"observation.images.{k}": dict(img) for k in CAM_MAP},
    }
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    ds = LeRobotDataset.create(repo_id=args.repo_id or f"local/alphabot2_stage1_{args.split}", fps=10,
                               features=features, root=args.out, robot_type="aibot2_isaac_public_asset",
                               use_videos=False, image_writer_threads=args.writer_threads)
    eps = sorted(d for s in args.src for d in Path(s, "episodes").glob("*") if (d / "meta.json").exists())
    used, skipped = [], []
    t0 = time.time()
    for d in eps:
        meta = json.loads((d / "meta.json").read_text())
        cid = meta["config"]["config_id"]
        if meta["config"]["split"] != args.split or (only is not None and cid not in only):
            continue
        if not meta.get("accepted"):
            skipped.append({"config_id": cid, "why": "not accepted"})
            continue
        low = np.load(d / "lowdim.npz")
        ims = dict(np.load(d / "images.npz"))
        prop, act = low["proprio"], low["action"]
        for t in range(len(act)):
            frame = {"observation.state": P.dataset_state_from_proprio(prop[t]).astype(np.float32),
                     "action": act[t].astype(np.float32), "task": meta["instruction"]}
            for k, src in CAM_MAP.items():
                frame[f"observation.images.{k}"] = ims[src][t]
            ds.add_frame(frame)
        ds.save_episode()
        used.append({"episode_index": len(used), "config_id": cid, "frames": int(len(act)),
                     "family": meta["family"], "arm": meta["arm"], "object": meta["object"], "cells": meta["cells"],
                     "source": str(d)})
        if len(used) % 20 == 0:
            print(f"{len(used)} episodes, {time.time() - t0:.0f}s", flush=True)
    ds.finalize()
    out = {"root": args.out, "split": args.split, "episodes": used, "skipped": skipped,
           "frames": int(sum(u["frames"] for u in used)), "features": {k: v["shape"] for k, v in features.items()},
           "camera_map": CAM_MAP, "wall_s": round(time.time() - t0, 1)}
    Path(args.out, "stage1_episode_index.json").write_text(json.dumps(out, indent=1, default=str))
    print("###CONVERT###" + json.dumps({k: v for k, v in out.items() if k != "episodes"}, default=str), flush=True)


if __name__ == "__main__":
    main()
