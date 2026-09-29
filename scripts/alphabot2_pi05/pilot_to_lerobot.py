"""Convert successful pilot episodes to a local LeRobot v3 dataset for pi0.5 SFT (no training here).

Layout matches the public AlphaBot pi0.5 checkpoint wherever it can:
  observation.state      16-D, checkpoint convention (dataset_state_from_proprio)
  observation.images.*   camera_front <- cam_chest, camera_top <- cam_front (head),
                         camera_left <- cam_right_wrist, camera_right <- cam_left_wrist
  action                 18-D CartesianMobileAction18 (public 16 dims + v, omega)
  task                   the episode's language instruction
Extra (not a policy input): observation.proprio16 (canonical proprio).
Stage labels / events / object poses stay in the pilot metadata, never in features.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np

REPO = os.environ.get("LMM_REPO", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402
from lmm.embodiments.alphabot2.contract import PROPRIO_LABELS  # noqa: E402

CAM_MAP = {"camera_front": "cam_chest", "camera_top": "cam_front",
           "camera_left": "cam_right_wrist", "camera_right": "cam_left_wrist"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="/nix/yanjun/zts/alphabot2/pilot")
    ap.add_argument("--out", default="/nix/yanjun/zts/alphabot2/pilot_lerobot")
    ap.add_argument("--repo-id", default="local/alphabot2_sim_pilot")
    ap.add_argument("--videos", action="store_true", help="encode videos (default: PNG images)")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--writer-threads", type=int, default=8)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    img = {"dtype": "video" if args.videos else "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": list(P.STATE_NAMES)},
        "observation.proprio16": {"dtype": "float32", "shape": (16,), "names": list(PROPRIO_LABELS)},
        "action": {"dtype": "float32", "shape": (18,), "names": list(C.CARTESIAN_MOBILE_ACTION18_LABELS)},
        **{f"observation.images.{k}": dict(img) for k in CAM_MAP},
    }
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    import time
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=10, features=features, root=args.out,
                               robot_type="aibot2_isaac_public_asset", use_videos=args.videos,
                               image_writer_threads=args.writer_threads)
    used, skipped = [], []
    for d in sorted(Path(args.pilot).glob("ep*")):
        meta = json.loads((d / "meta.json").read_text())
        if not meta.get("success"):
            skipped.append(d.name)
            continue
        if args.max_episodes and len(used) >= args.max_episodes:
            break
        t0 = time.time()
        low = np.load(d / "lowdim.npz")
        ims = {k: v for k, v in np.load(d / "images.npz").items()}
        t1 = time.time()
        prop, act = low["proprio"], low["action"]
        for t in range(len(act)):
            frame = {"observation.state": P.dataset_state_from_proprio(prop[t]).astype(np.float32),
                     "observation.proprio16": prop[t].astype(np.float32),
                     "action": act[t].astype(np.float32),
                     "task": meta["instruction"]}
            for k, src in CAM_MAP.items():
                frame[f"observation.images.{k}"] = ims[src][t]
            ds.add_frame(frame)
        t2 = time.time()
        ds.save_episode()
        t3 = time.time()
        used.append({"episode": d.name, "frames": int(len(act)), "kind": meta["kind"], "arm": meta["arm"],
                     "load_s": round(t1 - t0, 1), "add_frames_s": round(t2 - t1, 1), "save_episode_s": round(t3 - t2, 1)})
        print("EPISODE", json.dumps(used[-1]), flush=True)
    ds.finalize()
    out = {"root": args.out, "episodes": used, "skipped_unsuccessful": skipped,
           "frames": int(sum(u["frames"] for u in used)), "features": {k: v["shape"] for k, v in features.items()}}
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
