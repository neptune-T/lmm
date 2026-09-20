"""Reward computation from verified terminal outcomes.

The optimizer consumes **generic verified trajectory groups**. Nothing in this
module knows about OVMM skills, place suffixes or semantic phases — that
coupling is exactly what the cleanup removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..schemas import VerificationRecord


@dataclass
class RewardConfig:
    """Terminal reward shaping.

    ``success_weight`` dominates by design: shaping terms break ties between
    unsuccessful continuations, they do not substitute for task success.
    """

    success_weight: float = 1.0
    shaping_weights: Mapping[str, float] = field(default_factory=dict)
    clip_low: float | None = None
    clip_high: float | None = None


def terminal_reward(
    success: bool, metrics: Mapping[str, float], config: RewardConfig
) -> float:
    """Scalar reward for one verified continuation."""
    r = config.success_weight if success else 0.0
    for key, weight in config.shaping_weights.items():
        if key in metrics:
            r += weight * float(metrics[key])
    if config.clip_low is not None:
        r = max(config.clip_low, r)
    if config.clip_high is not None:
        r = min(config.clip_high, r)
    return r


def reward_from_verification(
    record: VerificationRecord, config: RewardConfig
) -> float:
    """Reward from the **verified** outcome only.

    The cheap score is deliberately ignored: it selected the candidate, it does
    not score it. Using ``cheap_score`` here would let low-precision noise leak
    into the gradient.
    """
    return terminal_reward(record.verified_success, record.verified_metrics, config)
