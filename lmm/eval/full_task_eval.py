"""Full-task evaluation from episode initial states.

This is the primary evaluation for an end-to-end method, and it is mandatory:
the canonical policy executes the whole task, so a frontier-conditioned number
cannot stand in for it.

The evaluation runs the end-to-end VLA and **must not** use the legacy OVMM
semantic skill sequencer. :func:`evaluate` rejects records whose metadata shows
a skill sequencer was active.

Paired statistics follow the frozen protocol already used for the 20-round
result (``paper_closure/single_execution_eval/``): the cluster unit is
``root_id``/``episode_id``, engineering failures count as binary failures, and
the decision rule is stated up front.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..frontier.frontier_estimator import wilson_interval
from ..schemas import FullRolloutRecord


class EvalContractError(ValueError):
    """A record violated the end-to-end evaluation contract."""


@dataclass
class FullTaskEvalResult:
    """Task success from episode initial states."""

    n_episodes: int
    n_rollouts: int
    n_success: int
    success_rate: float
    ci_low: float
    ci_high: float
    policy_version: str
    precision_backend: str
    per_episode: dict[str, float] = field(default_factory=dict)
    mean_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_type": "full_task_from_initial_state",
            "n_episodes": self.n_episodes,
            "n_rollouts": self.n_rollouts,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "policy_version": self.policy_version,
            "precision_backend": self.precision_backend,
            "per_episode": self.per_episode,
            "mean_metrics": self.mean_metrics,
        }


def _assert_end_to_end(records: Sequence[FullRolloutRecord]) -> None:
    for r in records:
        for step in r.steps:
            meta = step.observation_meta if isinstance(step.observation_meta, dict) else {}
            if meta.get("skill_sequencer_active") or meta.get("curr_skill"):
                raise EvalContractError(
                    f"rollout {r.rollout_id!r} was produced with a semantic skill "
                    "sequencer active. Full-task evaluation must use the "
                    "end-to-end VLA."
                )


def evaluate(records: Iterable[FullRolloutRecord]) -> FullTaskEvalResult:
    """Aggregate end-to-end rollouts into a full-task success report."""
    records = list(records)
    if not records:
        raise ValueError("no rollouts to evaluate")
    _assert_end_to_end(records)

    versions = {r.policy_version for r in records}
    backends = {r.precision_backend for r in records}
    if len(versions) > 1 or len(backends) > 1:
        raise EvalContractError(
            f"cannot pool rollouts across policy versions {sorted(versions)} or "
            f"backends {sorted(backends)}"
        )

    by_ep: dict[str, list[FullRolloutRecord]] = defaultdict(list)
    for r in records:
        by_ep[r.episode_id].append(r)

    n_success = sum(1 for r in records if r.terminal_success)
    lo, hi = wilson_interval(n_success, len(records))

    metric_keys = {k for r in records for k in r.terminal_metrics}
    mean_metrics = {
        k: sum(r.terminal_metrics.get(k, 0.0) for r in records) / len(records)
        for k in sorted(metric_keys)
    }

    return FullTaskEvalResult(
        n_episodes=len(by_ep),
        n_rollouts=len(records),
        n_success=n_success,
        success_rate=n_success / len(records),
        ci_low=lo,
        ci_high=hi,
        policy_version=next(iter(versions)),
        precision_backend=next(iter(backends)),
        per_episode={
            eid: sum(1 for r in rs if r.terminal_success) / len(rs)
            for eid, rs in sorted(by_ep.items())
        },
        mean_metrics=mean_metrics,
    )


def paired_comparison(
    before: Sequence[FullRolloutRecord],
    after: Sequence[FullRolloutRecord],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Episode-clustered paired comparison of two checkpoints.

    Mirrors the frozen decision rule: improvement is claimed only when the
    success delta and its cluster-bootstrap CI both sit above zero.
    """
    a = evaluate(before)
    b = evaluate(after)
    shared = sorted(set(a.per_episode) & set(b.per_episode))
    if not shared:
        raise EvalContractError("no shared episodes; comparison is not paired")

    deltas = [b.per_episode[e] - a.per_episode[e] for e in shared]
    mean_delta = sum(deltas) / len(deltas)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    ci = (
        means[int(0.025 * n_bootstrap)],
        means[min(int(0.975 * n_bootstrap), n_bootstrap - 1)],
    )

    improved = mean_delta > 0 and ci[0] > 0
    return {
        "comparison": f"{b.policy_version}_vs_{a.policy_version}",
        "cluster_unit": "episode_id",
        "n_clusters": len(shared),
        "before_success_rate": a.success_rate,
        "after_success_rate": b.success_rate,
        "mean_delta": mean_delta,
        "cluster_bootstrap_95ci": list(ci),
        "improved_episodes": sum(1 for d in deltas if d > 0),
        "degraded_episodes": sum(1 for d in deltas if d < 0),
        "unchanged_episodes": sum(1 for d in deltas if d == 0),
        "decision_rule": (
            "success delta and cluster-bootstrap CI above zero; otherwise no "
            "retained improvement"
        ),
        "interpretation": "RETAINED_IMPROVEMENT" if improved else "NO_RETAINED_IMPROVEMENT",
    }
