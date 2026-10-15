"""Merge sharded OVMM conversions (ovmm_to_lerobot.py --shard k/N --no-stats) into one LeRobot dataset.

lerobot's aggregate_datasets concatenates data / videos / metadata in shard order; our sidecars are merged
the same way: frame_meta.npz (episode indices offset per shard; stage and scene ids remapped to unified name
lists, since each shard numbered them in its own encounter order) and ovmm_episode_index.json. The merged
unit order must equal the manifest split exactly (shards are contiguous blocks), frame counts must agree,
then exact global stats are computed once over the merged frames and frozen to the manifest (exact_stats.py).

    python merge_shards.py --shards S0 S1 ... --out DIR --manifest M.json --split train
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    from lerobot.datasets.aggregate import aggregate_datasets
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shards = [Path(s) for s in a.shards]
    aggregate_datasets([f"local/shard{k}" for k in range(len(shards))], "local/merged", roots=shards, aggr_root=Path(a.out))

    stage_names, scene_names, fm_out, episodes, ep_off = [], [], {}, [], 0
    for s in shards:
        fm = dict(np.load(s / "frame_meta.npz"))
        sn, cn = [str(x) for x in fm["stage_names"]], [str(x) for x in fm["scene_names"]]
        for n in sn:
            if n not in stage_names:
                stage_names.append(n)
        for n in cn:
            if n not in scene_names:
                scene_names.append(n)
        st = np.array([stage_names.index(sn[i]) for i in fm["stage"]])
        sc = np.array([scene_names.index(cn[i]) for i in fm["scene"]])
        part = {**{k: v for k, v in fm.items() if k not in ("stage_names", "scene_names")},
                "episode_index": np.asarray(fm["episode_index"]) + ep_off, "stage": st, "scene": sc}
        for k, v in part.items():
            fm_out.setdefault(k, []).append(np.asarray(v))
        idx = json.load(open(s / "ovmm_episode_index.json"))
        for e in idx["episodes"]:
            episodes.append({**e, "episode_index": e["episode_index"] + ep_off})
        ep_off += len(idx["episodes"])
    fm_out = {k: np.concatenate(v) for k, v in fm_out.items()}
    np.savez(Path(a.out) / "frame_meta.npz", **fm_out, stage_names=np.asarray(stage_names), scene_names=np.asarray(scene_names))
    m = json.load(open(a.manifest))
    units = [str(e["unit"]) for e in episodes]
    if units != list(m[a.split]):
        raise SystemExit(f"merged unit order != manifest {a.split} ({len(units)} vs {len(m[a.split])})")
    info = json.load(open(Path(a.out) / "meta" / "info.json"))
    if info["total_frames"] != len(fm_out["episode_index"]) or info["total_episodes"] != len(episodes):
        raise SystemExit(f"merged frames/episodes mismatch: info {info['total_frames']}/{info['total_episodes']} vs "
                         f"frame_meta {len(fm_out['episode_index'])}/{len(episodes)}")
    shutil.copy(shards[0] / "policy_io.json", Path(a.out) / "policy_io.json")
    first = json.load(open(shards[0] / "ovmm_episode_index.json"))
    stats_fix = exact_stats.apply(a.out, a.manifest, a.split, units)
    out = {**{k: first[k] for k in ("split", "val_permille", "keep_settle", "policy_io", "camera_map", "video", "features") if k in first},
           "root": a.out, "episodes": episodes, "frames": int(len(fm_out["episode_index"])), "manifest": a.manifest,
           "dataset_version": m.get("dataset_version"), "shards": [str(s) for s in shards], "exact_stats": True,
           "lerobot_aggregate_quantile_error": stats_fix}
    (Path(a.out) / "ovmm_episode_index.json").write_text(json.dumps(out, indent=1, default=str))
    print("###MERGE###" + json.dumps({"episodes": len(episodes), "frames": out["frames"], "shards": len(shards)}), flush=True)


if __name__ == "__main__":
    main()
