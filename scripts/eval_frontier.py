#!/usr/bin/env python3
"""Stage 8a — frontier continuation success on train and held-out roots.

Reports success from restored frontier roots. This is NOT task success and
must never be reported as such; see eval_full_task.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO_ROOT  # noqa: F401  (path setup)

from lmm.eval.frontier_eval import evaluate, held_out_comparison
from lmm.schemas import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-probes", required=True)
    p.add_argument("--held-out-probes", default=None)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    train = evaluate(read_jsonl(args.train_probes), split="train")
    report: dict[str, object] = {"train": train.to_dict()}
    print(f"[train] {train.n_success}/{train.n_probes} over {train.n_roots} roots "
          f"= {train.success_rate:.3f} CI [{train.ci_low:.3f}, {train.ci_high:.3f}]")

    if args.held_out_probes:
        held = evaluate(read_jsonl(args.held_out_probes), split="held_out")
        report["held_out"] = held.to_dict()
        report["comparison"] = held_out_comparison(train, held)
        print(f"[held-out] {held.n_success}/{held.n_probes} = {held.success_rate:.3f}")
        print(f"[gap] {report['comparison']['delta']:+.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
