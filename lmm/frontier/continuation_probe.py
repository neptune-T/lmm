"""Branched stochastic continuations from a restored root.

One backend, two roles (this is deliberate — see the method doc):

``probe``    small ``K_probe`` sample used to estimate ``V_hat`` during
             frontier discovery.
``explore``  large ``K_cheap`` sample drawn from an already-selected frontier
             root to find promising continuations.

They differ only in ``K`` and in the ``mode`` tag written to each record. The
rollout path is identical, so a discovery estimate and an exploration draw are
never quietly produced by different code.

Determinism contract
--------------------
``docs/FRESH_POLICY_PROCESS_FIRST_DIVERGENCE_AUDIT_V1.md`` recorded
``MODEL_FORWARD_BACKEND_DIVERGENCE``: identical inputs in two cold policy
processes produced 274/320 differing float32 outputs. Continuations are
therefore only comparable when served by **one persistent policy process**.
:class:`ContinuationProbeRunner` takes an already-constructed policy handle and
never re-creates it, and each continuation gets an explicit distinct seed so
intended stochasticity is separable from backend noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from ..rollout.precision_backend import resolve
from ..schemas import ProbeRecord, SnapshotRecord


class ContinuationBackend(Protocol):
    """Executes one continuation from a restored snapshot.

    Implementations live in ``lmm.rollout.runner`` (real simulator) and in the
    tests (deterministic fake). Keeping this a Protocol is what lets the whole
    frontier pipeline be tested without a GPU.
    """

    def run_continuation(
        self,
        *,
        snapshot: SnapshotRecord,
        seed: int,
        precision_backend: str,
        policy_version: str,
    ) -> tuple[bool, dict[str, float], int, str]:
        """Return ``(success, metrics, trajectory_length, trajectory_path)``."""


@dataclass
class ProbeConfig:
    """Probe/explore budget. Values come from ``configs/frontier/``."""

    k_probe: int = 4
    k_cheap: int = 32
    probe_backend: str = "fp8_storage_bf16_compute"
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.k_probe < 2:
            raise ValueError(
                "k_probe must be >= 2: a frontier cannot be inferred from a "
                "single continuation"
            )
        resolve(self.probe_backend)  # fail fast on an unknown backend name


def probe_id_for(root_id: str, mode: str, index: int, seed: int) -> str:
    """Deterministic, collision-free probe identity.

    Encodes root, mode, index and seed, so identity never has to be recovered
    by parsing a filename.
    """
    return f"{root_id}::{mode}::{index:04d}::seed{seed}"


class ContinuationProbeRunner:
    """Draws stochastic continuations from restored roots."""

    def __init__(
        self,
        backend: ContinuationBackend,
        config: ProbeConfig,
        *,
        seed_fn: Callable[[int], int] | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self._seed_fn = seed_fn or (lambda i: self.config.base_seed + i)

    def _run(
        self,
        snapshot: SnapshotRecord,
        policy_version: str,
        k: int,
        mode: str,
        precision_backend: str,
    ) -> list[ProbeRecord]:
        records: list[ProbeRecord] = []
        for i in range(k):
            seed = self._seed_fn(i)
            success, metrics, length, path = self.backend.run_continuation(
                snapshot=snapshot,
                seed=seed,
                precision_backend=precision_backend,
                policy_version=policy_version,
            )
            records.append(
                ProbeRecord(
                    root_id=snapshot.root_id,
                    probe_id=probe_id_for(snapshot.root_id, mode, i, seed),
                    precision_backend=precision_backend,
                    seed=seed,
                    policy_version=policy_version,
                    mode=mode,
                    terminal_success=success,
                    terminal_metrics=metrics,
                    trajectory_length=length,
                    trajectory_path=path,
                )
            )
        return records

    def probe(
        self,
        snapshot: SnapshotRecord,
        policy_version: str,
        *,
        k: int | None = None,
        precision_backend: str | None = None,
    ) -> list[ProbeRecord]:
        """Discovery mode: small sample for estimating ``V_hat``."""
        return self._run(
            snapshot,
            policy_version,
            k or self.config.k_probe,
            "probe",
            precision_backend or self.config.probe_backend,
        )

    def explore(
        self,
        snapshot: SnapshotRecord,
        policy_version: str,
        *,
        k: int | None = None,
        precision_backend: str | None = None,
    ) -> list[ProbeRecord]:
        """Exploration mode: scaled sample from a selected frontier root."""
        return self._run(
            snapshot,
            policy_version,
            k or self.config.k_cheap,
            "explore",
            precision_backend or self.config.probe_backend,
        )

    def probe_many(
        self, snapshots: Sequence[SnapshotRecord], policy_version: str
    ) -> list[ProbeRecord]:
        """Probe a whole coarse candidate scan."""
        out: list[ProbeRecord] = []
        for snap in snapshots:
            out.extend(self.probe(snap, policy_version))
        return out
