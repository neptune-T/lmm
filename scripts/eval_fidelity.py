#!/usr/bin/env python3
"""Stage 8c — cheap vs BF16 agreement (question Q3).

Measures whether cheap rollout preserves enough candidate ordering that BF16
verification recovers genuinely useful trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO_ROOT  # noqa: F401  (path setup)

from lmm.eval.fidelity_eval import evaluate
from lmm.schemas import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verifications", required=True)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    result = evaluate(list(read_jsonl(args.verifications)), top_k=args.top_k)
    print(f"[fidelity] n={result.n} spearman={result.spearman_rho:+.3f} "
          f"kendall={result.kendall_tau:+.3f} "
          f"top{result.top_k}_retention={result.top_k_retention:.3f}")
    print(f"[rates] cheap={result.cheap_success_rate:.3f} "
          f"verified={result.verified_success_rate:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True),
                           encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
