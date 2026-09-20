"""High-fidelity verification of cheaply-discovered candidates.

Cheap rollout is a *search* device, never ground truth. Candidates selected
under a cheap backend are re-executed with the canonical high-fidelity policy
(BF16 pi0.5), and the two outcomes are stored side by side with full lineage:

    source_root -> cheap_rollout_id -> cheap_score -> selection_rank
                -> verification_rollout_id -> verified_success

:class:`~lmm.schemas.VerificationRecord` rejects a record whose cheap and
verification rollout ids are equal, so a cheap run can never be relabelled as
its own verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..rollout.precision_backend import HIGH_FIDELITY_BACKEND, require_high_fidelity
from ..schemas import CheapExplorationRecord, SnapshotRecord, VerificationRecord


class VerificationBackend(Protocol):
    """Re-executes a candidate continuation at high fidelity."""

    def rerun(
        self,
        *,
        snapshot: SnapshotRecord,
        seed: int,
        precision_backend: str,
        policy_version: str,
    ) -> tuple[str, bool, dict[str, float]]:
        """Return ``(verification_rollout_id, success, metrics)``."""


@dataclass
class VerifierConfig:
    """Verification budget and selection rule."""

    k_verify: int = 8
    backend: str = HIGH_FIDELITY_BACKEND
    select_by: str = "cheap_score"

    def __post_init__(self) -> None:
        require_high_fidelity(self.backend)
        if self.select_by not in ("cheap_score", "cheap_outcome"):
            raise ValueError(f"unknown select_by {self.select_by!r}")


def rank_candidates(
    candidates: Sequence[CheapExplorationRecord], config: VerifierConfig
) -> list[CheapExplorationRecord]:
    """Rank cheap candidates best-first, then truncate to ``k_verify``."""
    if config.select_by == "cheap_outcome":
        key = lambda c: (not c.cheap_outcome, -c.cheap_score)  # noqa: E731
    else:
        key = lambda c: -c.cheap_score  # noqa: E731
    return sorted(candidates, key=key)[: config.k_verify]


class BF16Verifier:
    """Re-runs top cheap candidates at high fidelity and records lineage."""

    def __init__(self, backend: VerificationBackend, config: VerifierConfig) -> None:
        self.backend = backend
        self.config = config

    def verify(
        self,
        snapshot: SnapshotRecord,
        candidates: Sequence[CheapExplorationRecord],
        policy_version: str,
    ) -> list[VerificationRecord]:
        """Verify the top-ranked cheap candidates from one root."""
        foreign = [c.root_id for c in candidates if c.root_id != snapshot.root_id]
        if foreign:
            raise ValueError(
                f"candidates from other roots {set(foreign)} cannot be verified "
                f"against snapshot {snapshot.root_id!r}"
            )

        out: list[VerificationRecord] = []
        for rank, cand in enumerate(rank_candidates(candidates, self.config)):
            rollout_id, success, metrics = self.backend.rerun(
                snapshot=snapshot,
                seed=cand.seed,
                precision_backend=self.config.backend,
                policy_version=policy_version,
            )
            out.append(
                VerificationRecord(
                    source_root=snapshot.root_id,
                    cheap_rollout_id=cand.rollout_id,
                    cheap_score=cand.cheap_score,
                    cheap_outcome=cand.cheap_outcome,
                    selection_rank=rank,
                    verification_rollout_id=rollout_id,
                    verification_backend=self.config.backend,
                    verified_success=success,
                    verified_metrics=metrics,
                    policy_version=policy_version,
                )
            )
        return out
