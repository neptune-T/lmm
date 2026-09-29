"""Write the Stage-1 task configs and the split manifest (numpy only; no Isaac).

    python make_configs.py OUT_DIR [--smoke]

train  -> expert demonstrations (SFT train set; the ONLY source of norm stats)
val    -> expert demonstrations for validation loss + the H_exec comparison
test   -> initial configurations for held-out evaluation (the expert runs on
          them only to measure solvability; its test trajectories are never
          converted to training data)
"""
import json, os, sys
from pathlib import Path

REPO = os.environ.get("LMM_REPO", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import stage1_tasks as T  # noqa: E402

SEED = 20260929
#: attempts per family (successes needed ~= 0.87 x attempts after expert failures)
PLAN = {
    "train": {"nav_pick_lift": 92, "nav_pick_nav_place": 127, "short_pick_place": 69, "reorient_pick_lift": 69},
    "val": {f: 10 for f in T.FAMILIES},
    "test": {f: 30 for f in T.FAMILIES},
}
SMOKE = {"train": {f: 2 for f in T.FAMILIES}}


def main():
    out = Path(sys.argv[1])
    plan = SMOKE if "--smoke" in sys.argv else PLAN
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "heldout_combinations": {k: [list(c) for c in v] for k, v in T.HELDOUT.items()},
                "factor_levels": {"base": T.BASE_LEVELS, "object": T.OBJECT_LEVELS, "receptacle": T.RECEPTACLE_LEVELS},
                "cells": {s: [list(c) for c in T.cells_of(s)] for s in T.SPLITS}, "plan": plan, "configs": {}}
    for split, per in plan.items():
        cfgs = [c.to_dict() for c in T.generate(split, per, SEED)]
        (out / f"configs_{split}.json").write_text(json.dumps(cfgs, indent=1))
        manifest["configs"][split] = [c["config_id"] for c in cfgs]
        print(split, len(cfgs))
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
