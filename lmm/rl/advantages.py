"""Group-relative advantages with auditable root identity.

GRPO normalises rewards **within a group of siblings sharing a root**. That
grouping is the scientific content of the algorithm, so this module:

* requires every sample to carry ``root_id`` and ``group_id`` explicitly;
* refuses to build a group spanning more than one root;
* never infers grouping from a filename or an ordering assumption.

Cross-root aggregation is allowed at the optimizer *batch* level — after
normalisation — and :func:`normalize_groups` keeps the per-root statistics so
that step remains auditable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..schemas import TrainingSampleRecord


class GroupingError(ValueError):
    """Sibling grouping was ambiguous or spanned unrelated roots."""


@dataclass
class AdvantageConfig:
    """How group-relative advantages are computed."""

    normalize_std: bool = True
    std_epsilon: float = 1e-6
    drop_degenerate_groups: bool = True
    min_group_size: int = 2


@dataclass
class GroupStats:
    """Per-group normalisation statistics, retained for audit."""

    group_id: str
    root_id: str
    size: int
    mean_reward: float
    std_reward: float
    degenerate: bool


def group_samples(
    samples: Iterable[TrainingSampleRecord],
) -> dict[str, list[TrainingSampleRecord]]:
    """Bucket samples by ``group_id``, asserting single-root groups."""
    groups: dict[str, list[TrainingSampleRecord]] = defaultdict(list)
    for s in samples:
        if not s.group_id:
            raise GroupingError(f"sample {s.rollout_id!r} has no group_id")
        if not s.root_id:
            raise GroupingError(f"sample {s.rollout_id!r} has no root_id")
        groups[s.group_id].append(s)

    for gid, members in groups.items():
        roots = {m.root_id for m in members}
        if len(roots) > 1:
            raise GroupingError(
                f"group {gid!r} spans multiple roots {sorted(roots)}. Siblings "
                "from unrelated roots must not share a group."
            )
    return dict(groups)


def normalize_groups(
    samples: Sequence[TrainingSampleRecord], config: AdvantageConfig
) -> tuple[list[TrainingSampleRecord], list[GroupStats]]:
    """Compute group-relative advantages.

    Returns ``(samples_with_advantages, per_group_stats)``. A group whose
    rewards are all equal carries no contrastive signal; it is reported as
    ``degenerate`` and dropped when ``drop_degenerate_groups`` is set.

    This is precisely the case the frozen TRAIN K=8 pilot hit: 64/64 siblings
    failed, every group was degenerate, and no gradient signal existed. That
    outcome is surfaced here rather than silently producing zero advantages.
    """
    groups = group_samples(samples)
    out: list[TrainingSampleRecord] = []
    stats: list[GroupStats] = []

    for gid in sorted(groups):
        members = groups[gid]
        rewards = [m.reward for m in members]
        n = len(rewards)
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        std = var**0.5
        degenerate = n < config.min_group_size or std <= config.std_epsilon

        stats.append(
            GroupStats(
                group_id=gid,
                root_id=members[0].root_id,
                size=n,
                mean_reward=mean,
                std_reward=std,
                degenerate=degenerate,
            )
        )

        if degenerate and config.drop_degenerate_groups:
            continue

        denom = (std + config.std_epsilon) if config.normalize_std else 1.0
        for m in members:
            out.append(
                TrainingSampleRecord.from_dict(
                    {**m.to_dict(), "advantage": (m.reward - mean) / denom}
                )
            )

    return out, stats


def degenerate_fraction(stats: Sequence[GroupStats]) -> float:
    """Fraction of groups with no contrastive signal.

    A value of 1.0 reproduces ``BRANCH_SUPPORT_NOT_OBSERVED`` and means the
    root bank cannot support an online update.
    """
    if not stats:
        return 0.0
    return sum(1 for s in stats if s.degenerate) / len(stats)
