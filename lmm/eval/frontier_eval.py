"""Evaluation at frontier roots.

Reports continuation success from restored roots, for the training roots and
for a held-out root set. Kept strictly separate from
:mod:`lmm.eval.full_task_eval`: success from a frontier root is *not* task
success, and the historical codebase's habit of reporting only "success from
pre-place root" is exactly what this split prevents.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..frontier.frontier_estimator import wilson_interval
from ..schemas import ProbeRecord


@dataclass
class FrontierEvalResult:
    """Continuation success over a set of roots."""

    split: str
    n_roots: int
    n_probes: int
    n_success: int
    success_rate: float
    ci_low: float
    ci_high: float
    policy_version: str
    precision_backend: str
    per_root: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_type": "frontier_continuation",
            "split": self.split,
            "n_roots": self.n_roots,
            "n_probes": self.n_probes,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "policy_version": self.policy_version,
            "precision_backend": self.precision_backend,
            "per_root": self.per_root,
        }


def evaluate(
    probes: Iterable[ProbeRecord], *, split: str = "train"
) -> FrontierEvalResult:
    """Aggregate continuation probes into a frontier-success report."""
    probes = list(probes)
    if not probes:
        raise ValueError("no probes to evaluate")

    backends = {p.precision_backend for p in probes}
    versions = {p.policy_version for p in probes}
    if len(backends) > 1 or len(versions) > 1:
        raise ValueError(
            f"cannot pool probes across backends {sorted(backends)} or policy "
            f"versions {sorted(versions)}"
        )

    by_root: dict[str, list[ProbeRecord]] = defaultdict(list)
    for p in probes:
        by_root[p.root_id].append(p)

    n_success = sum(1 for p in probes if p.terminal_success)
    lo, hi = wilson_interval(n_success, len(probes))
    return FrontierEvalResult(
        split=split,
        n_roots=len(by_root),
        n_probes=len(probes),
        n_success=n_success,
        success_rate=n_success / len(probes),
        ci_low=lo,
        ci_high=hi,
        policy_version=next(iter(versions)),
        precision_backend=next(iter(backends)),
        per_root={
            rid: sum(1 for p in rs if p.terminal_success) / len(rs)
            for rid, rs in sorted(by_root.items())
        },
    )


def held_out_comparison(
    train: FrontierEvalResult, held_out: FrontierEvalResult
) -> dict[str, object]:
    """Generalisation gap between training and held-out frontier roots (Q5)."""
    return {
        "comparison": "held_out_vs_train_frontier",
        "train_success_rate": train.success_rate,
        "held_out_success_rate": held_out.success_rate,
        "delta": held_out.success_rate - train.success_rate,
        "train_ci": [train.ci_low, train.ci_high],
        "held_out_ci": [held_out.ci_low, held_out.ci_high],
        "overlapping_ci": not (
            held_out.ci_low > train.ci_high or train.ci_low > held_out.ci_high
        ),
    }
