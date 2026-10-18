"""Job plan for perturb_eval.py (deterministic, written before any rollout).

Start states: expert navigate_to_object states of (a) training-set eval units (the policy trained on them) and
(b) held-out v2 SFT-val episodes, sorted, first N_TRAIN / N_HELDOUT that qualify. Per unit two states: the 30 %
and 60 % quantiles of the navigate_to_object frames that are >= MIN_FAR_M from the expert's stop point (end of
navigate_to_object), so a 1 m lateral push stays far-NAV. Levels: none; pos 0.25 / 0.5 / 1.0 m; yaw 15 / 30 / 60
deg. Seeds 0, 1. Every (state, seed) gets every level (matched).
    perturb_plan.py --train-units FILE --heldout-index V2_VAL/ovmm_episode_index.json --out jobs.json
"""
import argparse, json
from pathlib import Path

import numpy as np

LEVELS = [{"name": "none", "kind": "none", "mag": 0.0},
          {"name": "pos0.25", "kind": "pos", "mag": 0.25}, {"name": "pos0.5", "kind": "pos", "mag": 0.5},
          {"name": "pos1.0", "kind": "pos", "mag": 1.0},
          {"name": "yaw15", "kind": "yaw", "mag": 15.0}, {"name": "yaw30", "kind": "yaw", "mag": 30.0},
          {"name": "yaw60", "kind": "yaw", "mag": 60.0}]


def states(src, min_far, quantiles):
    md = np.load(Path(src) / "metadata.npz")
    nav = [t for t, s in enumerate(str(x) for x in md["stage"]) if s == "navigate_to_object"]
    if len(nav) < 10:
        return []
    xz = md["base_xz_yaw"][:, :2]; stop = xz[nav[-1]]
    far = [t for t in nav if np.linalg.norm(xz[t] - stop) >= min_far]
    if len(far) < 10:
        return []
    return [{"t": int(far[int(q * (len(far) - 1))]), "q": q, "dist_to_stop_m": round(float(np.linalg.norm(xz[far[int(q * (len(far) - 1))]] - stop)), 2)}
            for q in quantiles]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-units", required=True)
    ap.add_argument("--heldout-index", required=True)
    ap.add_argument("--collect", default="/workspace/runtime/runs/collect_clean_v1")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--min-far", type=float, default=3.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    quant = (0.3, 0.6)
    pools = {"train_unit": sorted(u for u in open(a.train_units).read().replace("\n", ",").split(",") if u),
             "heldout": sorted(str(e["unit"]) for e in json.load(open(a.heldout_index))["episodes"])}
    want = {"train_unit": a.n_train, "heldout": a.n_heldout}
    picked, jobs = [], []
    for group, units in pools.items():
        n = 0
        for u in units:
            ep, rep = u.split("#")
            src = Path(a.collect) / "episodes" / f"{ep}_r{rep}"
            if not (src / "metadata.npz").exists():
                continue
            S = states(src, a.min_far, quant)
            if len(S) < len(quant):
                continue
            picked.append({"unit": u, "group": group, "source": str(src), "states": S})
            n += 1
            if n >= want[group]:
                break
    for p in picked:
        for s in p["states"]:
            for seed in a.seeds:
                for L in LEVELS:
                    jobs.append({"unit": p["unit"], "group": p["group"], "source": p["source"], "t": s["t"], "q": s["q"],
                                 "dist_to_stop_m": s["dist_to_stop_m"], "seed": seed, "level": L["name"], "level_spec": L})
    plan = {"split": "train", "levels": LEVELS, "seeds": a.seeds, "min_far_m": a.min_far, "quantiles": quant,
            "units": picked, "jobs": jobs}
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(json.dumps({"units": len(picked), "by_group": {g: sum(p["group"] == g for p in picked) for g in pools},
                      "states": sum(len(p["states"]) for p in picked), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
