"""Exact global normalization statistics for a converted AlphaBot 2 OVMM LeRobot dataset.

lerobot 0.6.1 aggregates quantiles as a frame-weighted mean of PER-EPISODE quantiles, which is not the
dataset quantile: measured on sanity_v1, the gripper q99 came out 0.48-0.52 (true 1.0) and arm dims
were off by 30-60 % of their range, so QUANTILES normalization would push much of the data far outside
[-1, 1]. The policy interface fixes "q01/q99 of the OVMM TRAIN data", so for observation.state and action
every statistic (min, max, mean, std, q01, q10, q50, q90, q99, count) is recomputed exactly over all
frames and written into meta/stats.json; lerobot's aggregate is kept in meta/stats_lerobot_aggregate.json
and meta/exact_stats.json records that (and how) the replacement was done.
Image statistics are untouched (VISUAL features use identity normalization). Idempotent.

    python exact_stats.py DATASET_ROOT [--manifest M.json]

With --manifest the stats are frozen to that dataset version: meta/exact_stats.json records the manifest
sha256, dataset_version, and that the dataset's episodes are exactly the manifest's split (else exit 1).
"""
import argparse, glob, hashlib, json, shutil
from pathlib import Path

import numpy as np

KEYS = ("observation.state", "action")
QS = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def frames(root, key):
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(f"{root}/data/*/*.parquet"))
    if not fs:
        raise SystemExit(f"{root}: no parquet data")
    return np.concatenate([np.stack(pq.read_table(f, columns=[key]).column(key).to_numpy(zero_copy_only=False))
                           for f in fs]).astype(np.float64)


def exact(a):
    s = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0), "count": np.array([len(a)])}
    s.update({k: np.quantile(a, q, axis=0) for k, q in QS.items()})
    return {k: v.tolist() for k, v in s.items()}


def apply(root, manifest=None, split=None, units=None):
    """units/split: the dataset's episodes (the converter passes them; else read ovmm_episode_index.json)."""
    root = Path(root)
    frozen = {}
    if manifest:
        m = json.loads(Path(manifest).read_text())
        if units is None:
            idx = json.loads((root / "ovmm_episode_index.json").read_text())
            units, split = [str(e["unit"]) for e in idx["episodes"]], idx.get("split")
        if units != m[split]:
            raise SystemExit(f"{root}: episodes are not exactly the manifest's {split} split")
        frozen = {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                  "dataset_version": m["dataset_version"], "split": split, "episodes": len(units),
                  "computed_from": f"exactly the frozen manifest's {split} episodes (all frames after settle truncation)"}
    sf = root / "meta" / "stats.json"
    stats = json.loads(sf.read_text())
    agg = root / "meta" / "stats_lerobot_aggregate.json"
    if not agg.exists():
        shutil.copy(sf, agg)
    aggregate = json.loads(agg.read_text())
    report = {}
    for key in KEYS:
        new = exact(frames(root, key))
        old = aggregate.get(key, {})
        rng = np.maximum(np.asarray(new["max"]) - np.asarray(new["min"]), 1e-9)
        if "q01" in old and "q99" in old:
            err = np.maximum(np.abs(np.asarray(old["q01"]) - new["q01"]), np.abs(np.asarray(old["q99"]) - new["q99"])) / rng
            report[key] = {"aggregate_q_rel_err_max": float(err.max()), "dim": int(err.argmax())}
        stats[key] = new
    tmp = sf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=1))
    tmp.replace(sf)
    (root / "meta" / "exact_stats.json").write_text(json.dumps(
        {"keys": list(KEYS), "method": "exact global statistics over all frames (exact_stats.py)",
         "frames": int(stats[KEYS[0]]["count"][0]), "replaced_aggregate_error": report,
         "stats_sha256": hashlib.sha256(sf.read_bytes()).hexdigest(), **frozen}, indent=1))
    print("###EXACT_STATS###" + json.dumps({"root": str(root), **report}), flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    apply(a.root, a.manifest)
