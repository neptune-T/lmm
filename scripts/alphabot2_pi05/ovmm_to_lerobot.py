"""Accepted AlphaBot 2 OVMM trajectories -> a local LeRobot dataset (one split per call).

Policy-visible features only (the frozen contract):
  observation.images.camera_top     <- head_rgb          (240x320)
  observation.images.camera_left    <- right_wrist_rgb   (checkpoint convention, as in stage1_to_lerobot.py)
  observation.images.camera_right   <- left_wrist_rgb
  observation.state                 16-D, checkpoint convention (dataset_state_from_proprio)
  action                            18-D CartesianMobileAction18
  task                              the OVMM-style instruction (no arm / stage token)
Nothing privileged is written into the dataset. Per-frame sampling metadata (stage, base command, arm,
scene, episode) goes to a SEPARATE file, frame_meta.npz, read only by the training sampler.

Split: a deterministic hold-out of whole OVMM episodes (all replicas of an episode on the same side):
crc32(episode) % 1000 < val_permille -> "val" (SFT validation loss only; never used for evaluation).
Only episodes the collector accepted (OVMM success + fresh-reset replay) are read.

    python ovmm_to_lerobot.py --src /workspace/runtime/runs/collect_v1 --split train --out DIR [--png]
"""
import argparse, glob, json, os, shutil, sys, time, zlib
from pathlib import Path

import numpy as np

REPO = os.environ.get("LMM_REPO", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402

CAM_MAP = IO.CAMERA_MAP                     # the one interface (crossed wrists; camera_front absent)


def split_of(episode, val_permille):
    return "val" if zlib.crc32(f"sft-split:{episode}".encode()) % 1000 < val_permille else "train"


def accepted(src):
    names = set(glob.glob(f"{src}/results*.jsonl")) | {f"{src}/results_w{i:02d}.jsonl" for i in range(100)}
    rows = {}
    for f in sorted(n for n in names if os.path.exists(n)):     # bosfs readdir lags: probe worker files too
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("accepted") and r.get("dir"):
                rows[str(r["episode_id"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="collection OUT dirs")
    ap.add_argument("--split", required=True, choices=("train", "val"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-permille", type=int, default=50)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all accepted")
    ap.add_argument("--keep-settle", type=int, default=5,
                    help="post-release 'settle' frames kept per episode (the OVMM stability window is not SFT behaviour)")
    ap.add_argument("--png", action="store_true", help="lossless PNG frames instead of video (millions of files)")
    ap.add_argument("--writer-threads", type=int, default=16)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    img = {"dtype": "image" if args.png else "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
    features = {"observation.state": {"dtype": "float32", "shape": (16,), "names": list(IO.STATE_LABELS)},
                "action": {"dtype": "float32", "shape": (18,), "names": list(IO.ACTION_LABELS)},
                **{f"observation.images.{k}": dict(img) for k in CAM_MAP}}
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    ds = LeRobotDataset.create(repo_id=f"local/alphabot2_ovmm_{args.split}", fps=10, features=features, root=args.out,
                               robot_type="alphabot2_habitat_ovmm", use_videos=not args.png,
                               image_writer_threads=args.writer_threads)
    rows = []
    for s in args.src:
        rows += [r for r in accepted(s).values() if split_of(r.get("episode") or r["episode_id"], args.val_permille) == args.split]
    rows.sort(key=lambda r: str(r["episode_id"]))
    if args.max_episodes:
        rows = rows[: args.max_episodes]
    stage_names, fm = {}, {k: [] for k in ("episode_index", "frame_index", "stage", "v", "omega", "arm", "scene")}
    scenes, used = {}, []
    t0 = time.time()
    for r in rows:
        d = Path(r["dir"])
        meta = json.loads((d / "meta.json").read_text())
        low = np.load(d / "lowdim.npz")
        ims = np.load(d / "images.npz")
        md = np.load(d / "metadata.npz")
        prop, act, stg = low["proprio"], low["action"], [str(x) for x in md["stage"]]
        T_raw = len(act)
        if not (len(prop) == T_raw == len(stg) and all(ims[c].shape[0] == T_raw for c in CAM_MAP.values())):
            raise RuntimeError(f"{d}: misaligned arrays")
        # the settle interval sits at the end (after retract); keep only its first --keep-settle frames
        settle = [i for i, x in enumerate(stg) if x == "settle"]
        T = (settle[0] + min(len(settle), args.keep_settle)) if settle else T_raw
        if settle and any(x != "settle" for x in stg[settle[0]:T_raw] if x != "done"):
            raise RuntimeError(f"{d}: non-settle frames after the settle interval")
        frames = {k: ims[c] for k, c in CAM_MAP.items()}
        for t in range(T):
            frame = {"observation.state": IO.state_from_proprio(prop[t]).astype(np.float32),
                     "action": act[t].astype(np.float32), "task": meta["instruction"]}
            for k in CAM_MAP:
                frame[f"observation.images.{k}"] = frames[k][t]
            ds.add_frame(frame)
        ds.save_episode()
        ei = len(used)
        sc = scenes.setdefault(meta["scene"], len(scenes))
        for t in range(T):
            fm["episode_index"].append(ei)
            fm["frame_index"].append(t)
            fm["stage"].append(stage_names.setdefault(stg[t], len(stage_names)))
            fm["v"].append(float(act[t][16]))
            fm["omega"].append(float(act[t][17]))
            fm["arm"].append(0 if meta.get("arm") == "left" else 1)
            fm["scene"].append(sc)
        used.append({"episode_index": ei, "unit": r["episode_id"], "episode": r.get("episode"), "replica": r.get("replica"),
                     "frames": T, "frames_raw": T_raw, "settle_frames_dropped": T_raw - T, "arm": meta.get("arm"), "scene": meta["scene"], "object": meta.get("object"),
                     "source": str(d)})
        if len(used) % 50 == 0:
            print(f"{len(used)}/{len(rows)} episodes, {time.time() - t0:.0f}s", flush=True)
    ds.finalize()
    np.savez(Path(args.out) / "frame_meta.npz", **{k: np.asarray(v) for k, v in fm.items()},
             stage_names=np.asarray(sorted(stage_names, key=stage_names.get)),
             scene_names=np.asarray(sorted(scenes, key=scenes.get)))
    Path(args.out, "policy_io.json").write_text(json.dumps(IO.describe(), indent=1))
    out = {"root": args.out, "split": args.split, "val_permille": args.val_permille, "episodes": used, "policy_io": IO.describe(),
           "frames": int(sum(u["frames"] for u in used)), "camera_map": CAM_MAP, "video": not args.png,
           "features": {k: v["shape"] for k, v in features.items()}, "wall_s": round(time.time() - t0, 1)}
    Path(args.out, "ovmm_episode_index.json").write_text(json.dumps(out, indent=1, default=str))
    print("###CONVERT###" + json.dumps({k: v for k, v in out.items() if k != "episodes"}, default=str), flush=True)


if __name__ == "__main__":
    main()
