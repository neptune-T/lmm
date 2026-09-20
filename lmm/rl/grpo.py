"""GRPO-style objective over verified sibling groups.

Interface cleanup only — this is not a new algorithm. It reproduces the
group-relative policy-optimization objective the project already used, with
two changes:

* it consumes :class:`~lmm.schemas.TrainingSampleRecord` (generic verified
  trajectory groups), not OVMM skill structures;
* the trust-region / KL guard is explicit, carrying over the intent of
  ``scripts/train/post_update_trust_region_guard.py``.

The gradient itself belongs to the policy backend; this module owns the
objective, the guard and the reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..schemas import TrainingSampleRecord


@dataclass
class GRPOConfig:
    """GRPO objective and trust-region parameters."""

    clip_ratio: float = 0.2
    kl_coefficient: float = 0.0
    max_approx_kl: float | None = 0.05
    learning_rate: float = 1e-5
    update_epochs: int = 1


@dataclass
class GRPOStepResult:
    """Outcome of evaluating the objective on one batch."""

    loss: float
    mean_advantage: float
    mean_reward: float
    approx_kl: float
    clip_fraction: float
    n_samples: int
    guard_tripped: bool
    guard_reason: str = ""


def clipped_surrogate(
    advantage: float, log_ratio: float, clip_ratio: float
) -> tuple[float, bool]:
    """PPO/GRPO clipped surrogate for one sample.

    Returns ``(objective, was_clipped)``. Maximising the objective is
    equivalent to minimising its negation, which is what the loss reports.
    """
    ratio = math.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = min(max(ratio, 1.0 - clip_ratio), 1.0 + clip_ratio) * advantage
    return (min(unclipped, clipped), clipped < unclipped)


def approx_kl_from_log_ratios(log_ratios: Sequence[float]) -> float:
    """Schulman's low-variance approximate KL: ``mean(exp(r) - 1 - r)``.

    Non-negative by construction, unlike ``-mean(log_ratio)``.
    """
    if not log_ratios:
        return 0.0
    return sum(math.exp(r) - 1.0 - r for r in log_ratios) / len(log_ratios)


def evaluate_batch(
    samples: Sequence[TrainingSampleRecord],
    log_ratios: Sequence[float],
    config: GRPOConfig,
) -> GRPOStepResult:
    """Evaluate the GRPO objective and the trust-region guard on a batch.

    ``log_ratios[i]`` is ``log pi_new(a_i|s_i) - log pi_old(a_i|s_i)`` for
    sample ``i``, supplied by the policy backend. Keeping it an argument is
    what makes the objective testable without a model.
    """
    if len(samples) != len(log_ratios):
        raise ValueError(
            f"samples ({len(samples)}) and log_ratios ({len(log_ratios)}) "
            "must align one-to-one"
        )
    if not samples:
        return GRPOStepResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, False, "empty_batch")

    objectives: list[float] = []
    n_clipped = 0
    for s, lr in zip(samples, log_ratios):
        obj, clipped = clipped_surrogate(s.advantage, lr, config.clip_ratio)
        objectives.append(obj)
        n_clipped += int(clipped)

    n = len(samples)
    kl = approx_kl_from_log_ratios(log_ratios)
    loss = -sum(objectives) / n + config.kl_coefficient * kl

    tripped, reason = False, ""
    if config.max_approx_kl is not None and kl > config.max_approx_kl:
        tripped = True
        reason = (
            f"approx_kl {kl:.6f} exceeds max_approx_kl {config.max_approx_kl}; "
            "update should be rejected and the checkpoint rolled back"
        )

    return GRPOStepResult(
        loss=loss,
        mean_advantage=sum(s.advantage for s in samples) / n,
        mean_reward=sum(s.reward for s in samples) / n,
        approx_kl=kl,
        clip_fraction=n_clipped / n,
        n_samples=n,
        guard_tripped=tripped,
        guard_reason=reason,
    )
