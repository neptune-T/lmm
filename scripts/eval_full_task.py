#!/usr/bin/env python3
"""Stage 8b — full-task success from episode initial states.

The primary evaluation for an end-to-end method. Rejects any rollout produced
with the legacy semantic skill sequencer active. Optionally runs the paired,
episode-clustered comparison against a baseline checkpoint using the same
decision rule as the frozen single-execution protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO_ROOT  # noqa: F401  (path setup)

from lmm.eval.full_task_eval import evaluate, paired_comparison
from lmm.schemas import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", required=True, help="end-to-end rollouts to score")
    p.add_argument("--baseline-rollouts", default=None,
                   help="baseline checkpoint rollouts for a paired comparison")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    records = list(read_jsonl(args.rollouts))
    result = evaluate(records)
    report: dict[str, object] = {"full_task": result.to_dict()}
    print(f"[full-task] {result.n_success}/{result.n_rollouts} over "
          f"{result.n_episodes} episodes = {result.success_rate:.4f} "
          f"CI [{result.ci_low:.3f}, {result.ci_high:.3f}]")

    if args.baseline_rollouts:
        baseline = list(read_jsonl(args.baseline_rollouts))
        cmp = paired_comparison(baseline, records, n_bootstrap=args.n_bootstrap)
        report["paired_comparison"] = cmp
        print(f"[paired] {cmp['comparison']}: delta {cmp['mean_delta']:+.4f} "
              f"CI {cmp['cluster_bootstrap_95ci']}")
        print(f"[verdict] {cmp['interpretation']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
