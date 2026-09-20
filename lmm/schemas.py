"""Canonical record schemas for the LMM pipeline.

Every scientific quantity the pipeline produces is one of the dataclasses
below. They are the *only* sanctioned way to move data between stages.

Two rules motivate this module:

1. **No filename parsing for scientific analysis.** Identity (``root_id``,
   ``group_id``, ``policy_version``) is carried inside the record. A record is
   valid on its own, wherever the file happens to sit.
2. **Cheap and high-fidelity outcomes are never mixed.** Every record that
   carries an outcome also carries the ``precision_backend`` that produced it,
   and :class:`VerificationRecord` keeps the two sides separate by construction.

See ``docs/DATA_FORMATS.md`` for the prose specification.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = "lmm_schema_v1"


class SchemaError(ValueError):
    """A record violated its schema contract."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class _Record:
    """Base for all records: round-trips to JSON and validates on creation."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:  # overridden by subclasses that need more
        for f in fields(self):
            if f.metadata.get("required") and not getattr(self, f.name):
                raise SchemaError(
                    f"{type(self).__name__}.{f.name} is required and must be non-empty"
                )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        d["record_type"] = type(self).__name__
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "_Record":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _req(**kw: Any) -> Any:
    return field(metadata={"required": True}, **kw)


# --------------------------------------------------------------------------
# Stage 1 — full-task rollout
# --------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One environment step of an end-to-end rollout.

    ``events`` are *annotations* produced by observers. They never influenced
    the action stored in this record.
    """

    step: int
    action: Sequence[float]
    observation_meta: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FullRolloutRecord(_Record):
    """A complete end-to-end trajectory from the episode's initial state."""

    rollout_id: str = _req(default="")
    episode_id: str = _req(default="")
    scene_id: str = _req(default="")
    task: str = _req(default="")
    policy_version: str = _req(default="")
    precision_backend: str = _req(default="")
    seed: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    terminal_success: bool = False
    terminal_metrics: dict[str, float] = field(default_factory=dict)
    created_utc: str = field(default_factory=_utc_now)

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)

    def __post_init__(self) -> None:
        # Rehydrate nested steps so a record read back from JSONL behaves
        # exactly like one built in memory.
        self.steps = [
            s if isinstance(s, StepRecord) else StepRecord(**s) for s in self.steps
        ]
        super().__post_init__()

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["steps"] = [s.to_dict() if isinstance(s, StepRecord) else s for s in self.steps]
        d["trajectory_length"] = self.trajectory_length
        return d


# --------------------------------------------------------------------------
# Stage 2 — candidate snapshots ("roots")
# --------------------------------------------------------------------------


@dataclass
class SnapshotRecord(_Record):
    """A restorable simulator state proposed as a branching root.

    ``candidate_reason`` records *why* this step was snapshotted
    (``fixed_interval`` / ``semantic_event`` / ``manual_debug`` / refinement),
    so a coarse-to-fine search remains auditable after the fact.
    """

    root_id: str = _req(default="")
    episode_id: str = _req(default="")
    scene_id: str = _req(default="")
    snapshot_step: int = 0
    snapshot_path: str = ""
    policy_version_that_generated_it: str = _req(default="")
    candidate_reason: str = _req(default="")
    source_rollout_id: str = ""
    created_utc: str = field(default_factory=_utc_now)


# --------------------------------------------------------------------------
# Stage 3 — continuation probes
# --------------------------------------------------------------------------


@dataclass
class ProbeRecord(_Record):
    """One stochastic continuation launched from a restored root.

    Used in both probe mode (estimating ``V_hat``) and explore mode (scaled
    branching). ``mode`` distinguishes them; the rollout backend is the same.
    """

    root_id: str = _req(default="")
    probe_id: str = _req(default="")
    precision_backend: str = _req(default="")
    seed: int = 0
    policy_version: str = _req(default="")
    mode: str = "probe"  # "probe" | "explore"
    terminal_success: bool = False
    terminal_metrics: dict[str, float] = field(default_factory=dict)
    trajectory_path: str = ""
    trajectory_length: int = 0
    created_utc: str = field(default_factory=_utc_now)

    def validate(self) -> None:
        super().validate()
        if self.mode not in ("probe", "explore"):
            raise SchemaError(f"mode must be 'probe' or 'explore', got {self.mode!r}")


# --------------------------------------------------------------------------
# Stage 4 — frontier summary
# --------------------------------------------------------------------------


@dataclass
class FrontierSummary(_Record):
    """Empirical continuation statistics for one root.

    ``success_rate`` is ``V_hat(root) ~= P_pi(task success | root)``.
    ``selected_as_frontier`` is set by a selector reading thresholds from
    config — never hard-coded here.
    """

    root_id: str = _req(default="")
    n_probes: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    uncertainty: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 1.0
    precision_backend: str = _req(default="")
    policy_version: str = _req(default="")
    selected_as_frontier: bool = False
    selection_reason: str = ""
    created_utc: str = field(default_factory=_utc_now)

    def validate(self) -> None:
        super().validate()
        if self.n_success > self.n_probes:
            raise SchemaError(
                f"n_success ({self.n_success}) exceeds n_probes ({self.n_probes})"
            )


# --------------------------------------------------------------------------
# Stage 5 — cheap exploration
# --------------------------------------------------------------------------


@dataclass
class CheapExplorationRecord(_Record):
    """A scaled cheap continuation, tagged with its sibling group."""

    root_id: str = _req(default="")
    group_id: str = _req(default="")
    rollout_id: str = _req(default="")
    precision_backend: str = _req(default="")
    seed: int = 0
    policy_version: str = _req(default="")
    cheap_score: float = 0.0
    cheap_outcome: bool = False
    trajectory_path: str = ""
    created_utc: str = field(default_factory=_utc_now)


# --------------------------------------------------------------------------
# Stage 6 — high-fidelity verification
# --------------------------------------------------------------------------


@dataclass
class VerificationRecord(_Record):
    """Links one cheap candidate to its independent high-fidelity re-run.

    The cheap prediction and the BF16 verification are stored in *separate
    fields*. Nothing in this pipeline may overwrite one with the other.
    """

    source_root: str = _req(default="")
    cheap_rollout_id: str = _req(default="")
    cheap_score: float = 0.0
    cheap_outcome: bool = False
    selection_rank: int = 0
    verification_rollout_id: str = _req(default="")
    verification_backend: str = _req(default="")
    verified_success: bool = False
    verified_metrics: dict[str, float] = field(default_factory=dict)
    policy_version: str = _req(default="")
    created_utc: str = field(default_factory=_utc_now)

    def validate(self) -> None:
        super().validate()
        if self.cheap_rollout_id == self.verification_rollout_id:
            raise SchemaError(
                "cheap and verification rollout ids must differ: verification "
                "must be an independent execution, not a relabelled cheap run"
            )


# --------------------------------------------------------------------------
# Stage 7 — RL training
# --------------------------------------------------------------------------


@dataclass
class TrainingSampleRecord(_Record):
    """One verified continuation as consumed by the optimizer.

    Root and group identity are mandatory and stored explicitly so that
    root-level advantage normalisation stays auditable after the fact.
    """

    root_id: str = _req(default="")
    group_id: str = _req(default="")
    rollout_id: str = _req(default="")
    policy_version: str = _req(default="")
    precision_backend: str = _req(default="")
    reward: float = 0.0
    advantage: float = 0.0
    trajectory_path: str = ""


@dataclass
class TrainingStepRecord(_Record):
    """One optimizer step, with checkpoint lineage."""

    policy_before: str = _req(default="")
    policy_after: str = _req(default="")
    optimizer_step: int = 0
    root_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)
    mean_reward: float = 0.0
    mean_advantage: float = 0.0
    loss: float = 0.0
    approx_kl: float | None = None
    created_utc: str = field(default_factory=_utc_now)


# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------

_RECORD_TYPES: dict[str, type] = {
    c.__name__: c
    for c in (
        FullRolloutRecord,
        SnapshotRecord,
        ProbeRecord,
        FrontierSummary,
        CheapExplorationRecord,
        VerificationRecord,
        TrainingSampleRecord,
        TrainingStepRecord,
    )
}


def write_jsonl(path: str | Path, records: Iterable[_Record]) -> int:
    """Append records to a JSONL file. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[_Record]:
    """Read records back, reconstructing their concrete type."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rtype = data.get("record_type")
            if rtype not in _RECORD_TYPES:
                raise SchemaError(f"unknown record_type {rtype!r} in {path}")
            yield _RECORD_TYPES[rtype].from_dict(data)
