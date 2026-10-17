#!/usr/bin/env python3
"""Exact sampler coverage of an SFT run: replays sft_ovmm.WeightedBatches (torch.multinomial over the
sampling.py frame weights, generator seeded seed * 1_000_003 + rank * 7919 + micro_step) for every micro-batch
the run trained on, and reports unique-frame coverage, repeat rate and per-stage effective epochs.
Reads only frame_meta.npz and the run's config (run_info_step000000.json); never touches a model.
    sampler_stats.py --run-info RUN/run_info_step000000.json [--run-info ...] --out stats.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampling  # noqa: E402


def replay(info, segments):
    """segments: [(first_step, last_step, world)] (1-based optimizer steps); each rank r draws micro-batches
    seeded seed * 1_000_003 + r * 7919 + micro_step, micro_step = (step - 1) * accum + j (resumes continue the stream)"""
    C = info["config"]
    fm = dict(np.load(Path(C["train_data"]) / "frame_meta.npz"))
    w, _ = sampling.weights(fm, C["sampling"])
    w = torch.as_tensor(w, dtype=torch.double)
    micro, accum = int(C["batch_per_gpu"]), int(C.get("grad_accum") or 1)
    counts = np.zeros(len(w), np.int64)
    for a, b, world in segments:
        for r in range(world):
            for s in range((a - 1) * accum, b * accum):
                g = torch.Generator().manual_seed(int(C["seed"]) * 1_000_003 + r * 7919 + s)
                np.add.at(counts, torch.multinomial(w, micro, replacement=True, generator=g).numpy(), 1)
    return fm, w.numpy(), counts


def report(fm, w, counts):
    names = [str(x) for x in fm["stage_names"]]
    st, ep = np.asarray(fm["stage"]), np.asarray(fm["episode_index"])
    draws, n = int(counts.sum()), len(counts)
    hit = counts > 0
    out = {"frames": n, "episodes": int(ep.max()) + 1, "draws": draws,
           "unique_frames_drawn": int(hit.sum()), "frame_coverage": round(float(hit.mean()), 4),
           "episode_coverage": round(float(len(np.unique(ep[hit])) / (int(ep.max()) + 1)), 4),
           "repeat_rate": round(1 - float(hit.sum()) / draws, 4),             # draws that hit an already-drawn frame
           "draws_per_drawn_frame_mean": round(draws / max(1, int(hit.sum())), 3), "max_draws_one_frame": int(counts.max()),
           "effective_epochs_overall": round(draws / n, 4), "stages": {}}
    for i, s in enumerate(names):
        m = st == i
        if not m.any():
            continue
        d = int(counts[m].sum())
        out["stages"][s] = {"frames": int(m.sum()), "frame_share": round(float(m.mean()), 4),
                            "draws": d, "draw_share": round(d / draws, 4),
                            "expected_draw_share": round(float(w[m].sum() / w.sum()), 4),
                            "effective_epochs": round(d / int(m.sum()), 4),
                            "frame_coverage": round(float(hit[m].mean()), 4),
                            "repeat_rate": round(1 - float(hit[m].sum()) / max(1, d), 4)}
    om = np.abs(np.asarray(fm["omega"])); nav = np.isin(st, [i for i, s in enumerate(names) if s.startswith("navigate")])
    if nav.any():
        for lab, sel in (("nav_turning_|omega|>0.05", nav & (om > 0.05)), ("nav_straight_|omega|<=0.05", nav & (om <= 0.05))):
            d = int(counts[sel].sum())
            out["stages"][lab] = {"frames": int(sel.sum()), "draws": d, "draw_share": round(d / draws, 4),
                                  "effective_epochs": round(d / max(1, int(sel.sum())), 4), "frame_coverage": round(float(hit[sel].mean()), 4)}
    arm = np.asarray(fm["arm"])
    out["arm_draw_share_left"] = round(float(counts[arm == 0].sum() / draws), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-info", nargs="+", required=True, help="RUN_INFO[@first-last:world,...] (default: all steps, world 1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep, seen = {}, {}
    for spec in a.run_info:
        p, _, seg = spec.partition("@")
        info = json.load(open(p)); C = info["config"]; name = C["name"]
        segments = [(int(x.split(":")[0].split("-")[0]), int(x.split(":")[0].split("-")[1]), int(x.split(":")[1]))
                    for x in seg.split(",")] if seg else [(1, int(C["steps"]), 1)]
        fm, w, counts = replay(info, segments)
        key = (C["train_data"], json.dumps(C["sampling"], sort_keys=True), C["seed"], C["batch_per_gpu"], C.get("grad_accum"), str(segments))
        rep[name] = {"run_info": p, "segments": segments, "train_data": C["train_data"], "steps": C["steps"],
                     "seed": C["seed"], "sampling": C["sampling"], **report(fm, w, counts)}
        if key in seen:
            rep[name]["identical_draws_to"] = seen[key][0]
            rep[name]["identical_draws_verified"] = bool(np.array_equal(seen[key][1], counts))
        else:
            seen[key] = (name, counts)
        print(name, json.dumps({k: v for k, v in rep[name].items() if k not in ("stages", "sampling")}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
