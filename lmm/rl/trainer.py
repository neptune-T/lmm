"""Online RL trainer: orchestration only.

Deliberately thin. The trainer sequences

    verified groups -> dataset -> advantages -> objective -> guarded update

and writes a :class:`~lmm.schemas.TrainingStepRecord` carrying the root and
group ids that produced each update. Rollout collection lives in
``lmm.rollout``; the trainer never collects data itself. That separation is
what the cleanup was for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..schemas import TrainingSampleRecord, TrainingStepRecord, VerificationRecord
from .advantages import AdvantageConfig, degenerate_fraction, normalize_groups
from .dataset_builder import DatasetConfig, build_dataset
from .grpo import GRPOConfig, GRPOStepResult, evaluate_batch
from .reward import RewardConfig


class PolicyBackend(Protocol):
    """Minimal contract a trainable policy must satisfy."""

    def log_ratios(self, samples: Sequence[TrainingSampleRecord]) -> Sequence[float]:
        """Log importance ratios of the current policy vs. the sampling policy."""

    def apply_update(
        self, samples: Sequence[TrainingSampleRecord], result: GRPOStepResult
    ) -> str:
        """Apply the update; return the new ``policy_version``."""

    def rollback(self, policy_version: str) -> None:
        """Restore a previous checkpoint after a rejected update."""


@dataclass
class TrainerConfig:
    """Everything the trainer needs, assembled from ``configs/rl/``."""

    reward: RewardConfig
    dataset: DatasetConfig
    advantage: AdvantageConfig
    grpo: GRPOConfig
    abort_on_all_degenerate: bool = True


@dataclass
class UpdateOutcome:
    """Result of one online update attempt."""

    applied: bool
    reason: str
    record: TrainingStepRecord | None
    result: GRPOStepResult | None
    degenerate_group_fraction: float


class OnlineTrainer:
    """One guarded online update per call."""

    def __init__(self, backend: PolicyBackend, config: TrainerConfig) -> None:
        self.backend = backend
        self.config = config

    def update(
        self,
        verifications: Sequence[VerificationRecord],
        *,
        policy_before: str,
        optimizer_step: int,
    ) -> UpdateOutcome:
        """Run one update from verified continuations."""
        samples = build_dataset(
            verifications, self.config.reward, self.config.dataset
        )
        if not samples:
            return UpdateOutcome(False, "no_samples", None, None, 0.0)

        normalized, stats = normalize_groups(samples, self.config.advantage)
        degenerate = degenerate_fraction(stats)

        if not normalized:
            reason = (
                "all_groups_degenerate: every sibling group had identical "
                "rewards, so there is no contrastive signal. This reproduces "
                "BRANCH_SUPPORT_NOT_OBSERVED and does not authorize an update."
            )
            if self.config.abort_on_all_degenerate:
                return UpdateOutcome(False, reason, None, None, degenerate)

        log_ratios = self.backend.log_ratios(normalized)
        result = evaluate_batch(normalized, log_ratios, self.config.grpo)

        if result.guard_tripped:
            return UpdateOutcome(
                False, f"trust_region_guard: {result.guard_reason}", None, result,
                degenerate,
            )

        policy_after = self.backend.apply_update(normalized, result)
        record = TrainingStepRecord(
            policy_before=policy_before,
            policy_after=policy_after,
            optimizer_step=optimizer_step,
            root_ids=sorted({s.root_id for s in normalized}),
            group_ids=sorted({s.group_id for s in normalized}),
            mean_reward=result.mean_reward,
            mean_advantage=result.mean_advantage,
            loss=result.loss,
            approx_kl=result.approx_kl,
        )
        return UpdateOutcome(True, "applied", record, result, degenerate)
