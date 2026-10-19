"""Job plan for manip_from_arrival.py (deterministic, written before any rollout).

Units: (a) v2 TRAIN units (the policy trained on these exact episodes), (b) v2 SFT-VAL units (held-out base
episodes, same OVMM train split). Sorted by unit id, at most one seed-bank replica per base episode, first N of
each group whose recorded expert episode has both entry stages. Entries per unit:
  obj    first frame of the expert's "reach" stage   = expert navigate_to_object finished (standoff reached and
         turned to the planned heading); nothing grasped yet
  recep  first frame of the expert's "reach_place"   = expert navigate_to_receptacle finished, object in hand
Modes per (unit, entry): "expert_rest" (replay of the expert's own remaining actions: the restored state can be
completed) and "policy" x seeds.
    manip_plan.py --manifest MANIFEST.json --val-index V2_VAL/ovmm_episode_index.json --out jobs.json
"""
import argparse, json
from pathlib import Path

import numpy as np

ENTRIES = {"obj": "reach", "recep": "reach_place"}


def entry_steps(src):
    md = np.load(Path(src) / "metadata.npz")
    st = [str(x) for x in md["stage"]]
    out = {}
    for e, s in ENTRIES.items():
        if s not in st:
            return None
        out[e] = st.index(s)
    out["n_frames"] = len(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-index", required=True)
    ap.add_argument("--collect", default="/workspace/runtime/runs/collect_clean_v1")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-heldout", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pools = {"train_unit": sorted(json.load(open(a.manifest))["train"]),
             "heldout": sorted(str(e["unit"]) for e in json.load(open(a.val_index))["episodes"])}
    want = {"train_unit": a.n_train, "heldout": a.n_heldout}
    units, jobs = [], []
    for group, pool in pools.items():
        seen = set()
        for u in pool:
            ep, rep = u.split("#")
            if ep in seen:
                continue
            src = Path(a.collect) / "episodes" / f"{ep}_r{rep}"
            if not (src / "metadata.npz").exists():
                continue
            es = entry_steps(src)
            if es is None:
                continue
            units.append({"unit": u, "group": group, "source": str(src), **es})
            seen.add(ep)
            if sum(x["group"] == group for x in units) >= want[group]:
                break
    for x in units:
        for e in ENTRIES:
            base = {"unit": x["unit"], "group": x["group"], "source": x["source"], "entry": e, "t": x[e]}
            jobs.append({**base, "mode": "expert_rest", "seed": -1})
            for s in a.seeds:
                jobs.append({**base, "mode": "policy", "seed": s})
    Path(a.out).write_text(json.dumps({"split": "train", "entries": ENTRIES, "seeds": a.seeds, "units": units,
                                       "jobs": jobs}, indent=1))
    print(json.dumps({"units": len(units), "by_group": {g: sum(x["group"] == g for x in units) for g in pools},
                      "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
